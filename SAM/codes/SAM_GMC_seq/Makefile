CC=g++
CPPFLAGS=-O3 -std=c++11 -fopenmp
EXTER=../MyMath.o ../SAM_Base.o ../myinclude/myutils_3.o

samgmc: SAM_GMC.o samgmc.cpp $(EXTER)
	$(CC) SAM_GMC.o samgmc.cpp $(EXTER) -o samgmc $(CPPFLAGS)
SAM_GMC.o: SAM_GMC.cpp SAM_GMC.hpp ../GMC.hpp $(EXTER)
	$(CC) -c SAM_GMC.cpp $(CPPFLAGS)

.PHONY: clean
clean:
	-rm *.o samgmc

