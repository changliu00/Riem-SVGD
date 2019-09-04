#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits> // numeric_limits
#include <sys/stat.h> // mkdir
#include "SAM_Base.hpp"
#include "MyMath.hpp"

/***********************************/
// SAM_Base
SAM_Base::SAM_Base(const ParamManager &pm):
	purpose(For::tr), pm(pm), V(0), D(0), K(pm.i("K")), Nmax(pm.i("Nmax")), thBnin(pm.i("thBnin")), thN(pm.i("thN")), 
	sigma(pm.d("sig")), kappa0(pm.d("kp0")), kappa1(pm.d("kp1"))
{
	LoadData(pm.s("tr_data"), data, const_cast<int&>(V), const_cast<int&>(D));
	// set M and alpha
	M.resize(V);
	M.setZero();
	for(const auto &xi: data) M += xi;
	M.normalize();
	alpha.resize(K);
	alpha.setConstant(pm.d("aph"));
	// initialize beta
	betaSamples.reserve(Nmax);
	betaSamples.emplace_back(V, K);
	MatrixXd &beta = betaSamples.back();
	Bystd::RandNormal randN;
	randN(beta);
	beta /= sqrt(V);
	beta.colwise() += M;
	beta.colwise().normalize();

	sampleTimes.reserve(Nmax);
	sampleTimes.emplace_back(0);
	MBar.resize(V);
	betaGram.resize(K, K);
	// prepare model name
	const_cast<string&>(dirname) = UniqueFilename(DatePrefix(pm.s("pfx")), "/");
	mkdir(dirname.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	// write settings(.m), M, alpha
	pm.Print(dirname+"settings", {"SAM_Base"});
	pm.Printm(dirname+"settings.m", {"SAM_Base"}, false, dirname.substr(0, dirname.size()-1)+"_settings");
	ofstream ofs(dirname+"M");
	ofs << scientific << setprecision(6);
	ofs << M.transpose() << endl;
	ofs.close(); ofs.open(dirname+"alpha");
	ofs << alpha.transpose() << endl;
}
SAM_Base::SAM_Base(For purpose, const string &dirname, const ParamManager &pm):
	purpose(purpose), dirname(dirname), pm(pm), V(0), D(0), K(0), Nmax(0), thBnin(0), thN(0)
{
	const_cast<ParamManager&>(pm).Clear();
	if( !const_cast<ParamManager&>(pm).Load(dirname+"settings") ) throw;
	ifstream ifs;
	int Ncur;
	ifs.open(dirname+"misc"); ifs >> Ncur >> const_cast<int&>(D) >> const_cast<int&>(V); ifs.close();
	const_cast<int&>(K) = pm.i("K"); const_cast<int&>(Nmax) = pm.i("Nmax");
	const_cast<int&>(thBnin) = pm.i("thBnin"); const_cast<int&>(thN) = pm.i("thN");
	sigma = pm.d("sig"); kappa0 = pm.d("kp0"); kappa1 = pm.d("kp1");

	switch(purpose) {
	case For::re: {
		LoadData(pm.s("tr_data"), data, const_cast<int&>(V), const_cast<int&>(D));
		betaSamples.reserve(Nmax);
		sampleTimes.reserve(Nmax);
		MBar.resize(V);
		betaGram.resize(K, K);
	} // no break;
	case For::ts: {
		ifs.open(dirname+"M"); 
		M.resize(V);
		for(int v=0; v<V; v++) ifs >> M(v);
		ifs.close();
		ifs.open(dirname+"alpha");
		alpha.resize(K);
		for(int k=0; k<K; k++) ifs >> alpha(k);
		ifs.close();
		ifs.open(dirname+"sampleTimes");
		sampleTimes.resize(Ncur);
		for(int n=0; n<Ncur; n++) ifs >> sampleTimes[n];
		ifs.close();
	} // no break;
	case For::tw: {
		betaSamples.resize(Ncur, MatrixXd(V, K));
		ifs.clear(); // "ifs" may be in bad state when reading "sampleTimes", since for "para" mode, "Ncur==Nmax" but usually "epochs<Ncur". Required even after "ifs.close()".
		ifs.open(dirname+"betaSamples");
		for(int n=0; n<Ncur; n++)
			for(int k=0; k<K; k++) {
				if(ifs.eof()) {
					cerr << "ERROR in SAM_Base::SAM_Base: the file \"betaSamples\" may be incomplete!" << endl; throw;
				}
				for(int v=0; v<V; v++) ifs >> betaSamples[n](v, k);
			}
		ifs.close();
	} break;
	default: {
		cerr << "Error in SAM_Base::SAM_Base: unknown purpose!" << endl; throw; 
	}
	}
}

void SAM_Base::WriteIncream(int pos)const
{
	const int Ncur = betaSamples.size();
	ofstream ofs;
	cout << "DO NOT KILL NOW! Writing to disk..." << flush;
	ofs.open(dirname+"betaSamples", ios::app);
	ofs << scientific << setprecision(6);
	for(int i=pos; i<Ncur; i++) ofs << betaSamples[i].transpose() << endl;
	ofs.close();
	ofs.open(dirname+"sampleTimes", ios::app);
	ofs << fixed << setprecision(3);
	for(int i=pos; i<Ncur; i++) ofs << sampleTimes[i] << endl;
	ofs.close();
	ofs.open(dirname+"misc", ios::trunc);
	ofs << Ncur << " " << D << " " << V << endl;
	ofs.close();
	// cout << "Done!" << endl; // This should be done in derived classes.
}
// SAM_Base
/********************************/

/********************************/
// SAM_Eval
SAM_Eval::SAM_Eval(const SAM_Base &master, const ParamManager &pm):
	type(Type::invalid), master(master), V(0), D(0), K(master.K), thN(pm.i("ts_thN")), bnin(numeric_limits<int>::max()), N(numeric_limits<int>::max()), perpRes(0)
{
	if(pm.s("ts_type") == "llog") const_cast<Type&>(type) = Type::llog;
	else if(pm.s("ts_type") == "flog") const_cast<Type&>(type) = Type::flog;
	else if(pm.s("ts_type") == "lbaylog") const_cast<Type&>(type) = Type::lbaylog;
	else if(pm.s("ts_type") == "fbaylog") const_cast<Type&>(type) = Type::fbaylog;
	else {
		cerr << "ERROR in SAM_Eval::SAM_Eval: unknown test type \"" << pm.s("ts_type") << "\"!" << endl; throw;
	}
	if(pm.Isname("ts_bnin")) bnin = pm.i("ts_bnin");
	if(pm.Isname("ts_N")) N = pm.i("ts_N");
	LoadData(pm.s("ts_data"), data, const_cast<int&>(V), const_cast<int&>(D));
	// prepare output dirname
	const_cast<string&>(dirname) = UniqueFilename(master.dirname+"logperp", "/");
	mkdir(dirname.data(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	pm.Print(dirname+"settings", {"SAM_Eval"});
	string varname = dirname; varname.pop_back();
	for(size_t pos = -1; (pos = varname.find('/', pos+1)) != string::npos; varname[pos] = '_');
	pm.Printm(dirname+"settings.m", {"SAM_Eval"}, false, varname+"_settings");
	ofstream ofs(dirname+"results.m");
	ofs << varname << " = [ ..." << endl;
	ofs << "];" << endl;
}
SAM_Eval::SAM_Eval(const SAM_Base &master, ParamManager &pm, const string &dirname):
	type(Type::invalid), master(master), dirname(dirname), V(0), D(0), K(master.K), thN(0), bnin(numeric_limits<int>::max()), N(numeric_limits<int>::max()), perpRes(0)
{
	if(master.purpose != SAM_Base::For::re) {cerr << "ERROR in SAM_Eval::SAM_Eval: \"master\" is not For::re!" << endl; throw;}
	if(!pm.Load(dirname+"settings")) throw;
	if(pm.s("ts_type") == "llog") const_cast<Type&>(type) = Type::llog;
	else if(pm.s("ts_type") == "flog") const_cast<Type&>(type) = Type::flog;
	else if(pm.s("ts_type") == "lbaylog") const_cast<Type&>(type) = Type::lbaylog;
	else if(pm.s("ts_type") == "fbaylog") const_cast<Type&>(type) = Type::fbaylog;
	else {
		cerr << "ERROR in SAM_Eval::SAM_Eval: unknown test type \"" << pm.s("ts_type") << "\"!" << endl; throw;
	}
	const_cast<int&>(thN) = pm.i("ts_thN"); 
	LoadData(pm.s("ts_data"), data, const_cast<int&>(V), const_cast<int&>(D));
	/* // load last "bnin" and "N"
	ifstream ifs(dirname+"results.m");
	if(ifs.fail()) {cerr << "ERROR in SAM_Eval::SAM_Eval: cannot load result file!" << endl; throw;}
	string buff; int pos1, pos2, pos3;
	pos2 = pos1 = ifs.tellg();
	while(!ifs.eof()) {
		pos3 = pos2; pos2 = pos1; pos1 = ifs.tellg();
		getline(ifs, buff);
	}
	ifs.clear(); ifs.seekg(pos3);
	ifs >> bnin >> N; */
}

double SAM_Eval::EvalPerpSeq(void)
{
	if( N == numeric_limits<int>::max() || bnin == numeric_limits<int>::max() ) {
		cerr << "ERROR in SAM_Eval::EvalPerpSeq: \"N\" or \"bnin\" are not deliberately set!" << endl; throw;
	}
	if(bnin < 0) {
		cerr << "ERROR in SAM_Eval::EvalPerpSeq: \"bnin\" should be non-negative!" << endl; throw;
	}
	if(N <= 0) N = master.betaSamples.size() - bnin;
	cout << "Evaluating log-perplexity with burn-in size " << bnin << " and test size " << N << " ..." << endl;
	if(bnin + N > master.betaSamples.size()) {
		cerr << "ERROR in SAM_Eval::EvalPerpSeq: There are only " << master.betaSamples.size() << " samples!" << endl; throw;
	}
	switch(type) {
	case Type::llog:
		this->EvalPerp_LiteLog(); break;
	case Type::flog:
		this->EvalPerp_FullLog(); break;
	case Type::lbaylog:
		this->EvalPerp_LiteBayesianLog(); break;
	case Type::fbaylog:
		this->EvalPerp_FullBayesianLog(); break;
	}
	cout << "Result: " << setprecision(9) << scientific << perpRes << endl;
	// write results
	cout << "DO NOT KILL NOW! Writing results to disk..." << flush;
	fstream fs(dirname+"results.m"); // Use "fstream" instead of "ofstream": the following line does not work for "ofstream"...
	fs.seekp(-3, ios::end); // Works for "fstream"; does not work for "ofstream"
	fs << bnin << " " << N << " ";
	fs << fixed << setprecision(3) << master.sampleTimes[bnin+N-1] << " " << scientific << setprecision(9) << perpRes << "; ..." << endl;
	fs << "];" << endl;
	fs.close();
	cout << "Done!" << endl;
	return perpRes;
}

double SAM_Eval::EvalPerpPara(void)
{
	bnin = 0; N = master.betaSamples.size();
	switch(type) {
	case Type::llog:
		this->EvalPerp_LiteLog(); break;
	case Type::flog:
		this->EvalPerp_FullLog(); break;
	case Type::lbaylog:
		this->EvalPerp_LiteBayesianLog(); break;
	case Type::fbaylog:
		this->EvalPerp_FullBayesianLog(); break;
	}
	cout << "Result: " << setprecision(9) << scientific << perpRes << endl;
	// write results
	fstream fs(dirname+"results.m");
	fs.seekp(-3, ios::end);
	fs << master.epochs << " ";
	fs << fixed << setprecision(3) << master.sampleTimes.back() << " " << scientific << setprecision(9) << perpRes << "; ..." << endl; // Use "sampleTimes.back()" since "sampleTimes" is a buffer: not all sample times are stored for the parallel versions!
	fs << "];" << endl;
	return perpRes;
}

//////////////////////////

void SAM_Eval::EvalPerp_LiteLog(void)
{
	double accumN, accumD = 0;

	MatrixXd beta(V, K);
	beta.setZero();
	for(int n=bnin; n<bnin+N; n++) beta += master.betaSamples[n];
	beta.colwise().normalize();

	Bystd::RandDir randdir(master.alpha);
	VectorXd theta(K);
	vector<VectorXd> BThetaDSamples(thN);
	for(auto &bi: BThetaDSamples){
		randdir(theta);
		bi.noalias() = beta * theta;
		bi.normalize();
	}

	ProgressBar pgb(D, 40, 4, '|', 'v');
	pgb.Reset();
	for(const auto &xi: data){
		accumN = -INFINITY;
		for(const auto &bi: BThetaDSamples)
			accumN = Byme::LogSum( accumN, master.kappa1 * bi.dot(xi) );
		accumD -= accumN;
		pgb();
	}
	perpRes = accumD/D + log(thN) - Byme::LogVmfC(V, master.kappa1);
}

void SAM_Eval::EvalPerp_FullLog(void)
{
	double accumN, accumD = 0;
	VectorXd theta(K), BThetaD(V);
	Bystd::RandDir randdir(master.alpha);

	MatrixXd beta(V, K);
	beta.setZero();
	for(int n=bnin; n<bnin+N; n++) beta += master.betaSamples[n];
	beta.colwise().normalize();

	ProgressBar pgb(D, 40, 4, '|', 'v');
	pgb.Reset();
	for(const auto &xi: data){
		accumN = -INFINITY;
		for(int n=0; n<thN; n++){
			randdir(theta);
			BThetaD.noalias() = beta * theta;
			BThetaD.normalize();
			accumN = Byme::LogSum( accumN, master.kappa1 * BThetaD.dot(xi) );
		}
		accumD -= accumN;
		pgb();
	}
	perpRes = accumD/D + log(thN) - Byme::LogVmfC(V, master.kappa1);
}

void SAM_Eval::EvalPerp_LiteBayesianLog(void)
{
	vector<VectorXd> thetaSamples(thN, VectorXd(K));
	vector<VectorXd> BThetaDSamples(thN, VectorXd(V));
	VectorXd accumsNBeta(D);
	accumsNBeta.fill(-INFINITY);

	Bystd::RandDir randdir(master.alpha);
	for(auto &thn: thetaSamples) randdir(thn);

	ProgressBar pgb(N, 40, 4, '|', 'v');
	pgb.Reset();
	for(int iBeta=bnin; iBeta<bnin+N; iBeta++){
		for(int n=0; n<thN; n++){
			BThetaDSamples[n].noalias() = master.betaSamples[iBeta] * thetaSamples[n];
			BThetaDSamples[n].normalize();
		}

		for(int d=0; d<D; d++)
			for(int n=0; n<thN; n++)
				accumsNBeta(d) = Byme::LogSum( accumsNBeta(d), master.kappa1 * BThetaDSamples[n].dot(data[d]) );
		pgb();
	}
	perpRes = -accumsNBeta.sum()/D + log(thN*N) - Byme::LogVmfC(V, master.kappa1);
}

void SAM_Eval::EvalPerp_FullBayesianLog(void)
{
	double accumNBeta, accumD = 0;
	VectorXd theta(K), BThetaD(V);
	Bystd::RandDir randdir(master.alpha);

	ProgressBar pgb(D, 40, 4, '|', 'v');
	pgb.Reset();
	for(const auto &xi: data){
		accumNBeta = -INFINITY;
		for(int iBeta=bnin; iBeta<bnin+N; iBeta++){
			for(int n=0; n<thN; n++){
				randdir(theta);
				BThetaD.noalias() = master.betaSamples[iBeta] * theta;
				BThetaD.normalize();
				accumNBeta = Byme::LogSum( accumNBeta, master.kappa1 * BThetaD.dot(xi) );
			}
		}
		accumD -= accumNBeta;
		pgb();
	}
	perpRes = accumD/D + log(thN*N) - Byme::LogVmfC(V, master.kappa1);
}
// SAM_Eval
/***********************************/

/***********************************/
// SAM_Topwords
SAM_Topwords::SAM_Topwords(const SAM_Base &master, const ParamManager &pm):
	master(master), V(master.V), K(master.K), topn(pm.i("tw_topn")), bnin(pm.i("tw_bnin")), N(pm.i("tw_N")), beta(V, K), dict(V)
{
	cout << "Reading dictionary \"" << pm.s("tw_dict") << "\"..." << flush;
	ifstream ifs(pm.s("tw_dict"));
	if( ifs.fail() ) {
		cerr << "ERROR in SAM_Topwords::SAM_Topwords: cannot open dictionary file!" << endl; throw;
	}
	for(auto &w: dict) ifs >> w;
	cout << "done!" << endl;
}

void SAM_Topwords::GetTopwords(void)
{
	// estimate mean topics
	beta.setZero();
	for(int n=bnin; n<bnin+N; n++) beta += master.betaSamples[n];
	// beta.colwise().normalize();
	// sort and write
	cout << "DO NOT KILL NOW!\nGenerating top " << topn << " words for " << master.K << " topics..." << endl;
	ofstream ofs(UniqueFilename(master.dirname+"topwords"));
	ofs << K << " " << topn << " " << bnin << " " << N << endl;
	int idx;
	ProgressBar pgb(K*topn, 40, 4, '|', 'v');
	pgb.Reset();
	for(int k=0; k<K; k++) {
		ofs << setw(3) << k;
		for(int n=0; n<topn; n++) {
			beta.col(k).maxCoeff(&idx);
			beta(idx, k) = -numeric_limits<double>::max();
			ofs << " " << dict[idx];
			pgb();
		}
		ofs << endl;
	}
	cout << "done!" << endl;
}
// SAM_Topwords
/***********************************/

void LoadData(const string &filename, vector<VectorXd> &data, int &V, int &D)
{
	ifstream ifs(filename);
	if( ifs.fail() ) {
		cerr << "ERROR in LoadData: data file name \"" << filename << "\" invalid!" << endl;
		throw (int)0;
	}
	int numTerms, termID, label;

	ifs >> V >> D;
	data.clear();
	data.resize(D, VectorXd::Zero(V));
	for(auto &xi: data){
		ifs >> numTerms >> label;
		for(int i=0; i<numTerms; i++){
			ifs >> termID;
			ifs.ignore(); // ignore one character ":"
			ifs >> xi(termID);
		}
	}
}

