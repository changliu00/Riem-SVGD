#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib> // std::srand, 
#include <ctime> // std::time
#include "SAM_GMC.hpp"
#include "../myinclude/myutils_3.hpp"
using namespace std;

void PrintUsage(void);
ParamManager pm;
int main(int argn, char *argv[])
{
	if(argn < 3) {
		PrintUsage(); return 1;
	}
	std::srand( unsigned(std::time(0)) );
	string command(argv[1]);

	if(command == "tr") {
		if( !pm.Load(argv[2]) || !pm.Adjust(argn, argv, 3) ) {
			PrintUsage(); return 1;
		}
		SAM_GMC model(pm);
		model.Sample();
	} else if(command == "ts") {
		if( (argn < 4) || !pm.Load(argv[3]) || !pm.Adjust(argn, argv, 4) ) {
			PrintUsage(); return 1;
		}
		const ParamManager mdpm;
		SAM_Base model(SAM_Base::For::ts, argv[2], mdpm);
		SAM_Eval tsModel(model, pm);
		const vector<int> &bnins = pm.ivec("ts_bnin");
		const vector<int> &Ns = pm.ivec("ts_N");
		for(const auto bnin: bnins)
			for(const auto N: Ns) {
				tsModel.bnin = bnin; tsModel.N = N; tsModel.EvalPerpSeq();
			}
	} else if(command == "tw") {
		if( (argn < 4) || !pm.Load(argv[3]) || !pm.Adjust(argn, argv, 4) ) {
			PrintUsage(); return 1;
		}
		const ParamManager mdpm;
		SAM_Base model(SAM_Base::For::tw, argv[2], mdpm);
		SAM_Topwords twModel(model, pm);
		const vector<int> &bnins = pm.ivec("tw_bnin");
		const vector<int> &Ns = pm.ivec("tw_N");
		for(const auto bnin: bnins)
			for(const auto N: Ns) {
				twModel.bnin = bnin; twModel.N = N; twModel.GetTopwords();
			}
	} else if(command == "re") {
		if(argn != 3) {
			PrintUsage(); return 1;
		}
		string dirname(argv[2]);
		size_t pos = dirname.find("logperp");
		if(pos == string::npos) {
			cerr << "ERROR in main: invalid \"tsDirname\"!" << endl; throw;
		}
		dirname.erase(pos);
		SAM_GMC model(dirname, pm);
		model.Sample(argv[2]);
	} else {
		PrintUsage(); return 1;
	}
	return 0;
}

void PrintUsage(void)
{
	cout << endl;
	cout << "Usage:" << endl;
	cout << "samgmc tr [settingsFilename] ([var]) ([val]) (...)" << endl;
	cout << "samgmc ts [modelDirname] [settingsFilename] ([var]) ([val]) (...)" << endl;
	cout << "samgmc tw [modelDirname] [settingsFilename] ([var]) ([val]) (...)" << endl;
	cout << "samgmc re [tsDirname]" << endl;
	cout << endl;
}

