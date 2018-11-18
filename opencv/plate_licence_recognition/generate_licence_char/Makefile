CPP=g++

plrchr:
	$(CPP) src/Main.cpp src/CvxText.h src/CvxText.cpp src/PlrGen.h src/PlrGen.cpp -o bin/plrchr -O2 -I /usr/include/freetype2 -I /usr/include/freetype2/freetype -lfreetype `pkg-config --cflags --libs opencv`

clean:
	rm bin/plrchr
