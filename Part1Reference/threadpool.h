// function prototypes
// this represents work that has to be 
// completed by a thread in the pool
typedef struct 
{
    // function should be replaced once client.c is worked on. 
    void (*function)(void *p);
    void *data;
}
task;

void execute(void (*somefunction)(void *p), void *p);
int pool_submit(void (*somefunction)(void *p), void *p);
void *worker(void *param);
void pool_init(void);
void pool_shutdown(void);