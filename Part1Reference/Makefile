# makefile for thread pool
#

CC=gcc
CFLAGS=-Wall
PTHREADS=-lpthread

all: mult generator

mult: client.o threadpool.o Queue.o matrix_generator.o sparse.o
	$(CC) $(CFLAGS) -o mult client.o threadpool.o Queue.o matrix_generator.o sparse.o $(PTHREADS)

generator: matrix_generator_main.o matrix_generator.o
	$(CC) $(CFLAGS) -o generator matrix_generator_main.o matrix_generator.o $(PTHREADS)

client.o: client.c matrix_generator.h
	$(CC) $(CFLAGS) -c client.c $(PTHREADS)

threadpool.o: threadpool.c threadpool.h
	$(CC) $(CFLAGS) -c threadpool.c $(PTHREADS)

Queue.o: Queue.c Queue.h
	$(CC) $(CFLAGS) -c Queue.c $(PTHREADS)

matrix_generator.o: matrix_generator.c matrix_generator.h
	$(CC) $(CFLAGS) -c matrix_generator.c $(PTHREADS)

matrix_generator_main.o: matrix_generator_main.c matrix_generator.h
	$(CC) $(CFLAGS) -c matrix_generator_main.c $(PTHREADS)

sparse.o: sparse.c sparse.h 
	$(CC) $(CFLAGS) -c sparse.c $(PTHREADS)

clean:
	rm -rf *.o
	rm -rf mult
	rm -rf generator
	rm -rf *.txt

