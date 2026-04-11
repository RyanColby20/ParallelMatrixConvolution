
typedef struct  node{
    task *data;
    struct node *next;
}node_t;

typedef struct {
    node_t *head;
    node_t *back;
}queue_t;

int add(queue_t *queue, node_t *newNode);
int pop(queue_t *queue);