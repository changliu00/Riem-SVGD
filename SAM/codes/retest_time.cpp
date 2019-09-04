#include <iostream>
#include <fstream>
#include <string> // atoi, std::string
#include <cstdlib> // std::srand, 
#include <ctime> // std::time
#include "SAM_Base.hpp"
#include "myinclude/myutils_3.hpp"
using namespace std;

void PrintUsage(void);
int main(int argc, char* argv[])
{
	if(argc < 3) {
		PrintUsage(); return 1;
	}
	int Nnew = atoi(argv[2]);
	if(Nnew == 0) {
		cerr << "ERROR in main: the content \"" << argv[2] << "\" is invalid for \"Nnew\"!" << endl;
		PrintUsage(); return 1;
	}
	ifstream ifs(argv[1]);
	if(ifs.fail()) {
		cerr << "ERROR in main: cannot open results file named \"" << argv[1] << "\"!" << endl;
		PrintUsage(); return 1;
	}
	ParamManager pm;
	if(argc == 3) {
		string tsSetFilename = argv[1];
		size_t pos = tsSetFilename.find("results.m");
		if(pos == string::npos) {
			cerr << "ERROR in main: invalid \"resultsFilename\": pattern \"results.m\" not found!" << endl; return 1;
		}
		tsSetFilename.erase(pos);
		tsSetFilename += "settings";
		if(!pm.Load(tsSetFilename)) {
			PrintUsage(); return 1;
		}
	} else {
		ifstream ifs(argv[3]);
		if(ifs.fail()) {
			string tsSetFilename = argv[1];
			size_t pos = tsSetFilename.find("results.m");
			if(pos == string::npos) {
				cerr << "ERROR in main: invalid \"resultsFilename\": pattern \"results.m\" not found!" << endl; return 1;
			}
			tsSetFilename.erase(pos);
			tsSetFilename += "settings";
			if( !pm.Load(tsSetFilename) || !pm.Adjust(argc, argv, 4) ) {
				PrintUsage(); return 1;
			}
		} else {
			ifs.close();
			if( !pm.Load(argv[3]) || !pm.Adjust(argc, argv, 4) ) {
				PrintUsage(); return 1;
			}
		}
	}

	std::srand( unsigned(std::time(0)) );
	string buff = argv[1];
	size_t pos = buff.find("logperp");
	if(pos == string::npos) {
		cerr << "ERROR in main: invalid \"resultsFilename\": pattern \"logperp\" not found!" << endl; return 1;
	}
	buff.erase(pos);
	const ParamManager mdpm;
	SAM_Base model(SAM_Base::For::ts, buff, mdpm);
	SAM_Eval tsModel(model, pm);

	int bninOld, Nold;
	getline(ifs, buff);
	ifs >> bninOld >> Nold;
	while(!ifs.fail()) {
		tsModel.bnin = bninOld + Nold - Nnew;
		if(tsModel.bnin < 0) tsModel.bnin = 0;
		tsModel.N = bninOld + Nold - tsModel.bnin;
		tsModel.EvalPerpSeq();
		getline(ifs, buff);
		ifs >> bninOld >> Nold;
	}
	return 0;
}

void PrintUsage(void)
{
	cout << endl;
	cout << "Usage:" << endl;
	cout << "retest_time [resultsFilename] [Nnew] ([var]) ([val]) (...)" << endl;
	cout << "retest_time [resultsFilename] [Nnew] [testSettingsFilename] ([var]) ([val]) (...)" << endl;
	cout << endl;
}

