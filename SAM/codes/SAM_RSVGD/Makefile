CC=g++
CPPFLAGS=-O3 -std=c++11 -fopenmp
EXTER=../MyMath.o ../SAM_Base.o ../myinclude/myutils_3.o

samrsvgd: SAM_RSVGD.o samrsvgd.cpp $(EXTER)
	$(CC) SAM_RSVGD.o samrsvgd.cpp $(EXTER) -o samrsvgd $(CPPFLAGS)
SAM_RSVGD.o: SAM_RSVGD.cpp SAM_RSVGD.hpp ../RSVGD_SphProd.hpp $(EXTER)
	$(CC) -c SAM_RSVGD.cpp $(CPPFLAGS)

.PHONY: clean
clean:
	-rm *.o samrsvgd

