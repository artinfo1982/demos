CPP=g++

plrchr:
	$(CPP) src/Main.cpp src/CvxText.h src/CvxText.cpp src/PlrChrGen.h src/PlrChrGen.cpp -o bin/plrchr -O2 -I /usr/include/freetype2 -I /usr/include/freetype2/freetype -lfreetype `pkg-config --cflags --libs opencv`

clean:
	rm bin/plrchr
