#pragma once
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <initializer_list> // std::initializer_list
#include <utility> // std::move

/**********************/
// ProgressBar
class ProgressBar
{
private:
	const int N, L, P;
	const char b, v, V;
	const bool headSensitive;
	const double TsksPerBar;
	
	int n, nNext;
	double nTsks;

public:
	explicit ProgressBar(int numTasks, int numPrint = 30, int numPartition = 2, char charBar = '|', char charLabel = 'v', char charEnds = 'V', bool headSensitive = false);
	void Reset(void);
	void operator()(void);
};
// ProgressBar
/**********************/


/**********************/
// parameter manager
//
// One entry of a settings file:
// [group_name]:
// [parameter_name] = [value1] ([value2]) (...)
// %: comment the rest in the line. No exceptions.
// The parameter names should not contain any whitespaces nor repeat for any group in the settings file. 
// Repeated parameter names are allowed for different types;
// An empty line separates groups. The default group name is "default"

template<class T>
struct Param
{
	std::vector<T> vals;
	bool isadj; // whether the parameter has been adjusted by command line (via Adjust) or by Reset

	Param(int n = 0): isadj(false) {}; // default constructor. The argument is just for "std::map::emplace" to construct an instance.
	Param(Param &&rhs): isadj(rhs.isadj), vals(std::move(rhs.vals)) {};

	bool Load(std::istringstream &iss) {
		// "iss" should not be ended up with whitespaces, so that if success, directly after extracting the last content, "iss" will be in "eof" state. 
		while( iss.good() ) {
			vals.emplace_back();
			iss >> vals.back();
		}
		return iss.eof();
	}
};

struct Group
{
	std::string gname;
	std::vector< std::map<std::string, Param<int>>::const_iterator >			giparams;
	std::vector< std::map<std::string, Param<double>>::const_iterator >			gdparams;
	std::vector< std::map<std::string, Param<std::string>>::const_iterator >	gsparams;

	Group(const std::string &gname): gname(gname) {};
	Group(std::string &&gname): gname(std::move(gname)) {};
	Group(Group &&rhs): gname(std::move(rhs.gname)), giparams(std::move(rhs.giparams)), 
		gdparams(std::move(rhs.gdparams)), gsparams(std::move(rhs.gsparams)) {};
	bool Empty(void)const { return ( giparams.empty() && gdparams.empty() && gsparams.empty() ); }
};

class ParamManager
{
private:
	std::vector<Group> groups;
	std::map< std::string, Param<int> >			iparams;
	std::map< std::string, Param<double> >		dparams;
	std::map< std::string, Param<std::string> >	sparams;

public:
	ParamManager() { groups.emplace_back("default"); }
	bool Load(const std::string &settingsFilename);
	bool Adjust(int argn, char *argv[], int startI = 1);
	void Clear(void) { groups.clear(); iparams.clear(); dparams.clear(); sparams.clear(); }
	
	// Get the content of the desired parameter. Exception thrown if no matching name. 
	int i(const std::string &paramName)const;
	double d(const std::string &paramName)const;
	const std::string& s(const std::string &paramName)const;

	// Access the desired std::vector of the parameter "paramName". Exception thrown if no matching name. 
	const std::vector<int>& ivec(const std::string &paramName)const;
	const std::vector<double>& dvec(const std::string &paramName)const;
	const std::vector<std::string>& svec(const std::string &paramName)const;

	// Reset the content of the desired parameter. Return whether there is matching name.
	bool Reset(const std::string &paramName, int iVal);
	bool Reset(const std::string &paramName, double dVal);
	bool Reset(const std::string &paramName, const char *cVal);

	// To see if a parameter has been adjusted by command line or by Reset. Exception thrown if no matching name. 
	bool Isadj(const std::string &paramName)const;
	
	bool Empty(void)const { return (iparams.empty() && dparams.empty() && sparams.empty()); }

	// Retrieve the desired parameter. Return whether there is matching name.
	bool Isname(const std::string &paramName)const { return (iparams.count(paramName) || dparams.count(paramName) || sparams.count(paramName)); }
	int NameCount(const std::string &paramName)const { return iparams.count(paramName) + dparams.count(paramName) + sparams.count(paramName); }

	// Print the contents of all the parameters in group
	void Print(std::ostream &out, std::initializer_list<std::string> sgroups = {})const; // If "sgroups" has a repeated group name, then the group will be printed twice.
	void Print(const std::string &filename = "", std::initializer_list<std::string> sgroups = {}, bool isapp = false)const; // If "filename==\"\"", print to the screen.
	void Printm(std::ostream &out, std::initializer_list<std::string> sgroups = {}, const std::string &varname = "temp_settings")const; // Print in an m-file format. If "sgroups" has a repeated group name, then the group will be printed twice.
	void Printm(const std::string &filename = "", std::initializer_list<std::string> sgroups = {}, bool isapp = false, const std::string &varname = "temp_settings")const; // Print in an m-file format. If "filename==\"\"", print to the screen.
};
// parameter manager
/**********************/

// Adjust the file name so that the name is different from the names of existing files
// Format: [stem]_[adjusted_repeat_number][suffix]
// char* UniqueFilename(char *filename, const char *stem, const char *sfx = NULL);
std::string& UniqueFilename(std::string &filename, const std::string &stem, const std::string &sfx = "");
std::string UniqueFilename(const std::string &stem, const std::string &sfx = "");

// For an identifier (string) based on time
// Format: [pfx][Mmm][dd]_[hh]
std::string& DatePrefix(std::string &str, const std::string &pfx = "z");
std::string DatePrefix(const std::string &pfx = "z");

