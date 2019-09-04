#include <iostream>
#include <iomanip>
#include <algorithm> // std::find_if
#include <cstdio> // sprintf and file's
#include <cstdlib> // atoi, atof
#include <cmath> // ceil, floor
#include <ctime> // time, localtime
#include "myutils_3.hpp"

/********************************/
// ProgressBar
ProgressBar::ProgressBar(int numTasks, int numPrint, int numPartition, char charBar, char charLabel, char charEnds, bool headSensitive):
	N(numTasks), L(numPrint), P(numPartition), b(charBar), v(charLabel), V(charEnds), headSensitive(headSensitive), TsksPerBar( (double)N/(double)L ) {};

void ProgressBar::Reset(void)
{
	int i;

	n = 0;
	if(!headSensitive){
		nTsks = TsksPerBar;
		nNext = (int)nTsks;
	}else{
		nTsks = 0;
		nNext = 0;
	}

	if(P+1 > L || P==1){
		std::cout << V;
		for(i=1; i<L-1; i++)
			std::cout << ' ';
		std::cout << V << std::endl;
	}else{
		double ratio = (double)L/(double)P;
		double Dpos = ratio;
		int pos = (int)Dpos;
		std::cout << V;
		for(i=1; i<L-1; i++){
			if(i == pos){
				std::cout << v;
				Dpos += ratio;
				pos = (int)Dpos;
			}else
				std::cout << ' ';
		}
		std::cout << V << std::endl;
	}
}

void ProgressBar::operator()(void)
{
	n++;
	if(N > L){
		if((!headSensitive && n == nNext) || (headSensitive && n > nNext))
		{
			std::cout << b << std::flush;
			nTsks += TsksPerBar;
			nNext = (int)nTsks;
		}
	}else{
		int incBars;
		if(!headSensitive){
			incBars = floor((double)(n*L)/(double)N) - floor((double)((n-1)*L)/(double)N);
		}else{
			incBars = ceil((double)(n*L)/(double)N) - ceil((double)((n-1)*L)/(double)N);
		}
		for(int i=0; i<incBars; i++){
			std::cout << b;
		}
		std::cout << std::flush;
	}
	if(n == N)
		std::cout << std::endl;
}
// ProgressBar
/***********************************/

/***********************************/
// ParamManager

bool ParamManager::Load(const std::string &settingsFilename)
{
	const std::string whitespaces(" \t\f\v\n\r");
	const std::string doublefeatures(".eE");
	const std::string numchars(whitespaces + doublefeatures + "1234567890+-");

	std::string linebuff, name;
	int pos, pivot, lineNum = 0;
	std::vector<Group>::iterator pgrp = groups.begin();
	std::ifstream ifs(settingsFilename);
	std::istringstream iss;

	if( ifs.fail() ){
		std::cerr << "ERROR in ParamManager::Load: cannot open settings file!" << std::endl;
		return false;
	}

	while( !ifs.eof() ) {
		lineNum++;
		// get a line
		std::getline(ifs, linebuff); // '\n' is extracted and not added in "linebuff" but discarded. Former contents in "linebuff" are discarded.
		// skip comment
		pos = linebuff.find('%');
		if(pos != std::string::npos) linebuff.erase(pos); // erase contents at and after "pos"
		// erase trailing whitespaces or ignore empty line 
		pos = linebuff.find_last_not_of(whitespaces);
		if(pos != std::string::npos) linebuff.erase(pos+1);
		else {
			pgrp = groups.begin();
			continue;
		}

		// classify the command of the line
		pivot = linebuff.find_first_of("=:");
		if(pivot == std::string::npos) {
			std::cerr << "WARNING in ParamManager::Load: line " << lineNum << " invalid and ignored! Neither '=' nor ':' appears!" << std::endl;
			continue;
		}
		// prepare name
		pos = linebuff.find_first_not_of(whitespaces);
		name = linebuff.substr(pos, pivot - pos);
		pos = name.find_last_not_of(whitespaces);
		if(pos != std::string::npos) name.erase(pos+1);
		else {
			std::cerr << "WARNING in ParamManager::Load: line " << lineNum << " does not provide a parameter name or a group name! The line ignored!" << std::endl;
			continue;
		}
		// parameter or group
		if(linebuff[pivot] == '=') { // parameter
			if(pivot == linebuff.size() - 1) { // no values provided for the parameter
				std::cerr << "WARNING in ParamManager::Load: no values provided for parameter \"" << name << "\" in line " << lineNum << "! Cannot determine the type of the parameter and it is ignored!" << std::endl;
				continue;
			}
			// load
			iss.clear();
			iss.str(linebuff.substr(pivot+1));
			if(linebuff.find_first_not_of(numchars, pivot+1) == std::string::npos) { // is a number
				if(linebuff.find_first_of(doublefeatures, pivot+1) == std::string::npos) { // is an int
					std::pair<std::map<std::string, Param<int>>::iterator, bool> ret = iparams.emplace(name, 0); // insert in-place if no repeated parameter names
					if(!ret.second) {
						std::cerr << "WARNING in ParamManager::Load: parameter \"" << name << "\" as type \"int\" already exists before line " << lineNum << "! The line ignored!" << std::endl;
						continue;
					} else {
						if( !ret.first->second.Load(iss) ) { // load
							iss.clear(); // reset iostate labels
							std::getline(iss, linebuff);
							std::cerr << "WARNING in ParamManager::Load: input stream error while loading parameter \"" << name << "\" in line " << lineNum << " as type \"int\"! Contents \"" << linebuff << "\" ignored!" << std::endl;
						}
						pgrp->giparams.emplace_back(ret.first);
					}
				} else { // is a double
					std::pair<std::map<std::string, Param<double>>::iterator, bool> ret = dparams.emplace(name, 0); // insert in-place if no repeated parameter names
					if(!ret.second) {
						std::cerr << "WARNING in ParamManager::Load: parameter \"" << name << "\" as type \"double\" already exists before line " << lineNum << "! The line ignored!" << std::endl;
						continue;
					} else {
						if( !ret.first->second.Load(iss) ) { // load
							iss.clear(); // reset iostate labels
							std::getline(iss, linebuff);
							std::cerr << "WARNING in ParamManager::Load: input stream error while loading parameter \"" << name << "\" in line " << lineNum << " as type \"double\"! Contents \"" << linebuff << "\" ignored!" << std::endl;
						}
						pgrp->gdparams.emplace_back(ret.first);
					}
				}
			} else { // is a string
				std::pair<std::map<std::string, Param<std::string>>::iterator, bool> ret = sparams.emplace(name, 0); // insert in-place if no repeated parameter names
				if(!ret.second) {
					std::cerr << "WARNING in ParamManager::Load: parameter \"" << name << "\" as type \"std::string\" already exists before line " << lineNum << "! The line ignored!" << std::endl;
					continue;
				} else {
					if( !ret.first->second.Load(iss) ) { // load
						iss.clear(); // reset iostate labels
						std::getline(iss, linebuff);
						std::cerr << "WARNING in ParamManager::Load: input stream error while loading parameter \"" << name << "\" in line " << lineNum << " as type \"std::string\"! Contents \"" << linebuff << "\" ignored!" << std::endl;
					}
					pgrp->gsparams.emplace_back(ret.first);
				}
			}
		} else { // group
			if(pivot < linebuff.size() - 1) std::cerr << "WARNING in ParamManager::Load: contents \"" << linebuff.substr(pivot+1) << "\" after ':' in line " << lineNum << " ignored!" << std::endl;
			pgrp = std::find_if(groups.begin(), groups.end(), [&name](const Group &grp){return grp.gname == name;});
			if(pgrp == groups.end()) {
				groups.emplace_back(name);
				pgrp = groups.end() - 1; // necessary!
			}
		}
	}
	return true;
}

bool ParamManager::Adjust(int argn, char *argv[], int startI)
{
	int i = startI;
	bool res;

	res = true;
	while(i < argn){
		/*
		if( iparams.count(argv[i]) ){
			i++;
			Param<int> *param = &(iparams.at(argv[i]));
			param->isadj = true;
			param->vals.clear();
			param->vals.emplace_back( std::atoi(argv[i]) );
		} else if( dparams.count(argv[i]) ){
			i++;
			Param<double> *param = &(dparams.at(argv[i]));
			param->isadj = true;
			param->vals.clear();
			param->vals.emplace_back( std::atof(argv[i]) );
		} else if( sparams.count(argv[i]) ){
			i++;
			Param<std::string> *param = &(sparams.at(argv[i]));
			param->isadj = true;
			param->vals.clear();
			param->vals.emplace_back( argv[i] );
		} else {
			res = false;
			std::cerr << "WARNING in ParamManager::Adjust: unknown parameter \"" << argv[i] << "\" ignored!" << std::endl;
		}
		*/
		try {
			Param<int> &param = iparams.at(argv[i]);
			param.isadj = true;
			param.vals.clear();
			i++;
			param.vals.emplace_back( std::atoi(argv[i]) );
		} catch (...) {
			try {
				Param<double> &param = dparams.at(argv[i]);
				param.isadj = true;
				param.vals.clear();
				i++;
				param.vals.emplace_back( std::atof(argv[i]) );
			} catch (...) {
				try {
					Param<std::string> &param = sparams.at(argv[i]);
					param.isadj = true;
					param.vals.clear();
					i++;
					param.vals.emplace_back( argv[i] );
				} catch (...) {
					res = false;
					std::cerr << "WARNING in ParamManager::Adjust: unknown parameter \"" << argv[i] << "\" ignored!" << std::endl;
				}
			}
		}
		i++;
	}
	return res;
}

int ParamManager::i(const std::string &paramName)const
{
	try {
		const Param<int> &param = iparams.at(paramName);
		if( param.vals.empty() ) {
			std::cerr << "ERROR in ParamManager::i: parameter \"" << paramName << "\" is now empty!" << std::endl;
			throw (int)0;
		} else {
			return param.vals.front();
		}
	} catch (const std::out_of_range &err) {
		std::cerr << "ERROR in ParamManager::i: parameter \"" << paramName << "\" not found as type \"int\"!" << std::endl;
		throw err;
	}
}
double ParamManager::d(const std::string &paramName)const
{
	try {
		const Param<double> &param = dparams.at(paramName);
		if( param.vals.empty() ) {
			std::cerr << "ERROR in ParamManager::d: parameter \"" << paramName << "\" is now empty!" << std::endl;
			throw (int)0;
		} else {
			return param.vals.front();
		}
	} catch (const std::out_of_range &err) {
		std::cerr << "ERROR in ParamManager::d: parameter \"" << paramName << "\" not found as type \"double\"!" << std::endl;
		throw err;
	}
}
const std::string& ParamManager::s(const std::string &paramName)const
{
	try {
		const Param<std::string> &param = sparams.at(paramName);
		if( param.vals.empty() ) {
			std::cerr << "ERROR in ParamManager::s: parameter \"" << paramName << "\" is now empty!" << std::endl;
			throw (int)0;
		} else {
			return param.vals.front();
		}
	} catch (const std::out_of_range &err) {
		std::cerr << "ERROR in ParamManager::s: parameter \"" << paramName << "\" not found as type \"std::string\"!" << std::endl;
		throw err;
	}
}

const std::vector<int>& ParamManager::ivec(const std::string &paramName)const
{
	try {
		return iparams.at(paramName).vals;
	} catch (const std::out_of_range &err) {
		std::cerr << "ERROR in ParamManager::ivec: parameter \"" << paramName << "\" not found as type \"int\"!" << std::endl;
		throw err;
	}
}
const std::vector<double>& ParamManager::dvec(const std::string &paramName)const
{
	try {
		return dparams.at(paramName).vals;
	} catch (const std::out_of_range &err) {
		std::cerr << "ERROR in ParamManager::dvec: parameter \"" << paramName << "\" not found as type \"double\"!" << std::endl;
		throw err;
	}
}
const std::vector<std::string>& ParamManager::svec(const std::string &paramName)const
{
	try {
		return sparams.at(paramName).vals;
	} catch (const std::out_of_range &err) {
		std::cerr << "ERROR in ParamManager::svec: parameter \"" << paramName << "\" not found as type \"std::string\"!" << std::endl;
		throw err;
	}
}

bool ParamManager::Reset(const std::string &paramName, int iVal)
{
	try {
		Param<int> &param = iparams.at(paramName);
		param.vals.clear();
		param.vals.emplace_back(iVal);
		param.isadj = true;
		return true;
	} catch (const std::out_of_range &err) {
		std::cerr << "WARNING in ParamManager::Reset(const std::string&, int): unknown parameter \"" << paramName << "\" ignored!" << std::endl;
		return false;
	}
}
bool ParamManager::Reset(const std::string &paramName, double dVal)
{
	try {
		Param<double> &param = dparams.at(paramName);
		param.vals.clear();
		param.vals.emplace_back(dVal);
		param.isadj = true;
		return true;
	} catch (const std::out_of_range &err) {
		std::cerr << "WARNING in ParamManager::Reset(const std::string&, double): unknown parameter \"" << paramName << "\" ignored!" << std::endl;
		return false;
	}
}
bool ParamManager::Reset(const std::string &paramName, const char *cVal)
{
	try {
		Param<std::string> &param = sparams.at(paramName);
		param.vals.clear();
		param.vals.emplace_back(cVal);
		param.isadj = true;
		return true;
	} catch (const std::out_of_range &err) {
		std::cerr << "WARNING in ParamManager::Reset(const std::string&, const char*): unknown parameter \"" << paramName << "\" ignored!" << std::endl;
		return false;
	}
}

bool ParamManager::Isadj(const std::string &paramName)const
{
	try {
		return iparams.at(paramName).isadj;
	} catch (...) {
		try {
			return dparams.at(paramName).isadj;
		} catch (...) {
			try {
				return sparams.at(paramName).isadj;
			} catch (const std::out_of_range &err) {
				std::cerr << "ERROR in ParamManager::Isadj: parameter \"" << paramName << "\" not found!" << std::endl;
				throw err;
			}
		}
	}
}

void ParamManager::Print(std::ostream &out, std::initializer_list<std::string> sgroups)const
{
	if(sgroups.size() == 0) {
		for(const auto &grp: groups) {
			if( !(grp.Empty()) ) {
				out << grp.gname << ":" << std::endl;
				for(const auto &ppair: grp.giparams) {
					out << ppair->first << " =";
					for(const auto &item: ppair->second.vals) out << " " << item;
					out << std::endl;
				}
				for(const auto &ppair: grp.gdparams) {
					out << ppair->first << " =";
					for(const auto &item: ppair->second.vals) out << " " << item;
					out << std::endl;
				}
				for(const auto &ppair: grp.gsparams) {
					out << ppair->first << " =";
					for(const auto &item: ppair->second.vals) out << " " << item;
					out << std::endl;
				}
				out << std::endl;
			}
		}
	} else {
		for(const auto &sgrp: sgroups) {
			std::vector<Group>::const_iterator pgrp = std::find_if(groups.cbegin(), groups.cend(), [&sgrp](const Group &grp){return (grp.gname == sgrp);});
			if(pgrp == groups.cend()) std::cerr << "WARNING in ParamManager::Print: group \"" << sgrp << "\" in the given list does not exist! The group ignored!" << std::endl;
			else {
				if( !(pgrp->Empty()) ) {
					out << pgrp->gname << ":" << std::endl;
					for(const auto &ppair: pgrp->giparams) {
						out << ppair->first << " =";
						for(const auto &item: ppair->second.vals) out << " " << item;
						out << std::endl;
					}
					for(const auto &ppair: pgrp->gdparams) {
						out << ppair->first << " =";
						for(const auto &item: ppair->second.vals) out << " " << item;
						out << std::endl;
					}
					for(const auto &ppair: pgrp->gsparams) {
						out << ppair->first << " =";
						for(const auto &item: ppair->second.vals) out << " " << item;
						out << std::endl;
					}
					out << std::endl;
				}
			}
		}
	}
}
void ParamManager::Print(const std::string &filename, std::initializer_list<std::string> sgroups, bool isapp)const
{
	if(filename.empty()) {
		std::cout << std::scientific;
		this->Print(std::cout, sgroups);
	} else {
		std::ofstream ofs;
		if(isapp) ofs.open(filename, std::ios::app);
		else ofs.open(filename, std::ios::trunc);
		ofs << std::scientific;
		this->Print(ofs, sgroups);
	}
}
void ParamManager::Printm(std::ostream &out, std::initializer_list<std::string> sgroups, const std::string &varname)const
{
	if(this->Empty()) {
		out << varname << " = struct();" << std::endl;
		return;
	}
	out << varname << " = struct( ..." << std::endl;
	if(sgroups.size() == 0) {
		for(const auto &grp: groups) {
			if( !(grp.Empty()) ) {
				out << "... % " << grp.gname << ":" << std::endl;
				for(const auto &ppair: grp.giparams) {
					out << "'" << ppair->first << "', [";
					for(const auto &item: ppair->second.vals) out << item << ", ";
					out.seekp(-2, std::ios_base::cur);
					out << "], ..." << std::endl;
				}
				for(const auto &ppair: grp.gdparams) {
					out << "'" << ppair->first << "', [";
					for(const auto &item: ppair->second.vals) out << item << ", ";
					out.seekp(-2, std::ios_base::cur);
					out << "], ..." << std::endl;
				}
				for(const auto &ppair: grp.gsparams) {
					out << "'" << ppair->first << "', [";
					for(const auto &item: ppair->second.vals) out << "'" << item << "', ' ', ";
					out.seekp(-7, std::ios_base::cur);
					out << "], ..." << std::endl;
				}
			}
		}
	} else {
		for(const auto &sgrp: sgroups) {
			std::vector<Group>::const_iterator pgrp = std::find_if(groups.cbegin(), groups.cend(), [&sgrp](const Group &grp){return (grp.gname == sgrp);});
			if(pgrp == groups.cend()) std::cerr << "WARNING in ParamManager::Print: group \"" << sgrp << "\" in the given list does not exist! The group ignored!" << std::endl;
			else {
				if( !(pgrp->Empty()) ) {
					out << "... % " << pgrp->gname << ":" << std::endl;
					for(const auto &ppair: pgrp->giparams) {
						out << "'" << ppair->first << "', [";
						for(const auto &item: ppair->second.vals) out << item << ", ";
						out.seekp(-2, std::ios_base::cur);
						out << "], ..." << std::endl;
					}
					for(const auto &ppair: pgrp->gdparams) {
						out << "'" << ppair->first << "', [";
						for(const auto &item: ppair->second.vals) out << item << ", ";
						out.seekp(-2, std::ios_base::cur);
						out << "], ..." << std::endl;
					}
					for(const auto &ppair: pgrp->gsparams) {
						out << "'" << ppair->first << "', [";
						for(const auto &item: ppair->second.vals) out << "'" << item << "', ' ', ";
						out.seekp(-7, std::ios_base::cur);
						out << "], ..." << std::endl;
					}
				}
			}
		}
	}
	out.seekp(-6, std::ios_base::cur);
	out << " ..." << std::endl;
	out << ");" << std::endl;
}
void ParamManager::Printm(const std::string &filename, std::initializer_list<std::string> sgroups, bool isapp, const std::string &varname)const
{
	if(filename.empty()) {
		std::cout << std::scientific;
		this->Printm(std::cout, sgroups, varname);
	} else {
		if(isapp) {
			std::fstream fs(filename); // enable "fs" to trace back by "seekp"
			fs.seekp(0, std::ios_base::end);
			fs << std::scientific;
			this->Printm(fs, sgroups, varname);
		} else {
			std::ofstream ofs(filename);
			ofs << std::scientific;
			this->Printm(ofs, sgroups, varname);
		}
	}
}
// ParamManager
/***********************************/

// Adjust the file name so that the name is different from the names of existing files
// Format: [stem]_[adjusted_repeat_number][suffix]
/*
char* UniqueFilename(char *filename, const char *stem, const char *sfx)
{
	FILE *fp = NULL;
	int iRep = 1;

	while(true){
		if(sfx != NULL)
			sprintf( filename, "%s_%d%s", stem, iRep, sfx );
		else
			sprintf( filename, "%s_%d", stem, iRep );
		fp = fopen(filename, "r");
		if(fp != NULL){
			fclose(fp);
			iRep++;
		}else{
			break;
		}
	}
	return filename;
}
*/
std::string& UniqueFilename(std::string &filename, const std::string &stem, const std::string &sfx)
{
	std::ifstream ifs;
	int iRep = 1;

	while(true){
		if(sfx.empty())
			filename = stem + "_" + std::to_string(iRep);
		else
			filename = stem + "_" + std::to_string(iRep) + sfx;
		ifs.open(filename.c_str());
		if(ifs.fail())
			break;
		else{
			ifs.close();
			iRep++;
		}
	}
	return filename;
}
std::string UniqueFilename(const std::string &stem, const std::string &sfx)
{
	std::string filename;
	std::ifstream ifs;
	int iRep = 1;

	while(true){
		if(sfx.empty())
			filename = stem + "_" + std::to_string(iRep);
		else
			filename = stem + "_" + std::to_string(iRep) + sfx;
		ifs.open(filename.c_str());
		if(ifs.fail())
			break;
		else{
			ifs.close();
			iRep++;
		}
	}
	return filename; // Maybe it is more efficient than "std::move(filename)" for a modern compiler.
}


// For an identifier (string) based on time
// Format: [pfx][Mmm][dd]_[hh]
std::string& DatePrefix(std::string &str, const std::string &pfx)
{
	char buff[9];
	std::time_t rawtime;
	struct std::tm *timeinfo;

	std::time(&rawtime);
	timeinfo = localtime(&rawtime);
	std::strftime( buff, 9, "%b%d_%H", timeinfo );
	// delete timeinfo;
	str = pfx + buff;
	return str;
}
std::string DatePrefix(const std::string &pfx)
{
	char buff[9];
	std::time_t rawtime;
	struct std::tm *timeinfo;

	std::time(&rawtime);
	timeinfo = localtime(&rawtime);
	std::strftime( buff, 9, "%b%d_%H", timeinfo );
	// delete timeinfo;
	return pfx + buff; // the returned string will be move-constructed by the rvalue returned by operator+ (which returns "string" but not "string&").
}


