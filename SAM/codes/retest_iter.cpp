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
	if(argc < 4) {
		PrintUsage(); return 1;
	}
	int Nnew, tsInterv, tsBegin;
	if((Nnew = atoi(argv[2])) == 0) {
		cerr << "ERROR in main: the content \"" << argv[2] << "\" is invalid for \"Nnew\"!" << endl; PrintUsage(); return 1;
	}
	if((tsInterv = atoi(argv[3])) == 0) {
		cerr << "ERROR in main: the content \"" << argv[3] << "\" is invalid for \"tsInterv\"!" << endl; PrintUsage(); return 1;
	}
	if(argc == 4) tsBegin = 2;
	else if((tsBegin = atoi(argv[4])) == 0) {
		cerr << "ERROR in main: the content \"" << argv[4] << "\" is invalid for \"tsBegin\"!" << endl; PrintUsage(); return 1;
	}
	ParamManager pm;
	if(argc < 6) {
		string tsSetFilename = argv[1];
		tsSetFilename += "settings";
		if(!pm.Load(tsSetFilename)) {
			PrintUsage(); return 1;
		}
	} else {
		ifstream ifs(argv[5]);
		if(ifs.fail()) {
			string tsSetFilename = argv[1];
			tsSetFilename += "settings";
			if( !pm.Load(tsSetFilename) || !pm.Adjust(argc, argv, 6) ) {
				PrintUsage(); return 1;
			}
		} else {
			ifs.close();
			if( !pm.Load(argv[5]) || !pm.Adjust(argc, argv, 6) ) {
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

	int checkpoint = tsBegin;
	while(checkpoint <= model.betaSamples.size()) {
		if(checkpoint < Nnew) {
			tsModel.bnin = 0; tsModel.N = checkpoint;
		} else {
			tsModel.bnin = checkpoint - Nnew; tsModel.N = Nnew;
		}
		tsModel.EvalPerpSeq();
		checkpoint += tsInterv;
	}
	return 0;
}

void PrintUsage(void)
{
	cout << endl;
	cout << "Usage:" << endl;
	cout << "retest_iter [evalDirname] [Nnew] [tsInterv] ([tsBegin]=2) ([var]) ([val]) (...)" << endl;
	cout << "retest_iter [evalDirname] [Nnew] [tsInterv] [tsBegin] [testSettingsFilename] ([var]) ([val]) (...)" << endl;
	cout << endl;
}

