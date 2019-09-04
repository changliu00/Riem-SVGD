CC=g++
CPPFLAGS=-O3 -std=c++11
CLEANED=myutils_3.o

.PHONY: all
all: myutils_3.o

myutils_3.o: myutils_3.cpp myutils_3.hpp
	$(CC) -c myutils_3.cpp $(CPPFLAGS)

.PHONY: clean
clean:
	-rm $(CLEANED)

