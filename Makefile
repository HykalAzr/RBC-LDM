all: whitedetect

CFLAGS=-fPIC -g -Wall 
LIBS = `pkg-config --libs opencv4`
INCLUDE = -I/usr/local/include/opencv4 -I/usr/include/libusb-1.0
FREE_LIBS = -L/usr/local/lib -lfreenect -lpthread

detection:  detection.cpp
	$(CXX) $(INCLUDE) ./serialLib/serialib.cpp $(CFLAGS) $? -o $@  $(LIBS) $(FREE_LIBS)

whitedetect:  whitedetect.cpp
	$(CXX) $(INCLUDE) ./serialLib/serialib.cpp $(CFLAGS) $? -o $@  $(LIBS) $(FREE_LIBS)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

clean:
	rm -rf *.o test

