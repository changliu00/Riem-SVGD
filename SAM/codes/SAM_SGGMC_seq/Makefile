CC=g++
CPPFLAGS=-O3 -std=c++11 -fopenmp
EXTER=../MyMath.o ../SAM_Base.o ../myinclude/myutils_3.o

samsggmc: SAM_SGGMC.o samsggmc.cpp $(EXTER)
	$(CC) SAM_SGGMC.o samsggmc.cpp $(EXTER) -o samsggmc $(CPPFLAGS)
SAM_SGGMC.o: SAM_SGGMC.cpp SAM_SGGMC.hpp ../GMC.hpp ../SGGMC.hpp $(EXTER)
	$(CC) -c SAM_SGGMC.cpp $(CPPFLAGS)

.PHONY: clean
clean:
	-rm *.o samsggmc

