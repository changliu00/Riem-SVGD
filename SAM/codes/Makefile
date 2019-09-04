CC=g++
CPPFLAGS=-O3 -std=c++11
EXTER=myinclude/myutils_3.o

.PHONY: all
all: MyMath.o SAM_Base.o retest_time retest_iter

SAM_Base.o: SAM_Base.hpp SAM_Base.cpp $(EXTER)
	$(CC) -c SAM_Base.cpp $(CPPFLAGS)
MyMath.o: MyMath.hpp MyMath.cpp
	$(CC) -c MyMath.cpp $(CPPFLAGS)
retest_time: retest_time.cpp SAM_Base.o MyMath.o $(EXTER)
	$(CC) retest_time.cpp SAM_Base.o MyMath.o $(EXTER) -o retest_time $(CPPFLAGS)
retest_iter: retest_iter.cpp SAM_Base.o MyMath.o $(EXTER)
	$(CC) retest_iter.cpp SAM_Base.o MyMath.o $(EXTER) -o retest_iter $(CPPFLAGS)

.PHONY: clean
clean:
	-rm *.o retest_time retest_iter

