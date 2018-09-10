all: dikin_walk.cpp dikin_walk.h
	gcc dikin_walk.c -o dikin_walk -Larmadillo/build -larmadillo

clean:
	rm -rf dikin_walk
