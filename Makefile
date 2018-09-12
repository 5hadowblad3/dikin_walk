all: dikin_walk.cpp dikin_walk.h
	g++ -g -std=c++11 dikin_walk.cpp -o dikin_walk.exe -Larmadillo/build -larmadillo -Iarmadillo/include -lopenblas -llapack -lglpk

clean:
	rm -rf dikin_walk
