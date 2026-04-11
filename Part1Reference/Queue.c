#include <stdio.h>
#include <stdlib.h>
#include "threadpool.h"

/*
This is a linked list implementation of a queue, behavior you would 
expect of a typical linked list is the same but you will also have a
pointer to the last node.
*/


typedef struct node
{
    task *data;
    struct node *next;
}
node_t;

typedef struct Queue
{
    node_t *head;
    node_t *back;
}
queue_t;

int add(queue_t *queue, node_t *newNode)
{
    
    // If the Queue is empty then the head and back are set to the node
    if (queue -> head == NULL){
        newNode -> next = NULL;
        queue -> head = newNode;
        queue -> back = newNode;
        return 0;
    }

    newNode -> next = NULL;
    queue -> back -> next = newNode;
    queue -> back = newNode;
    
    return 0;
}

// Will only delete first node
int pop(queue_t *queue){

    node_t *nodeDel;
    nodeDel = queue -> head;
    
    if (nodeDel == NULL){
        printf("Queue is empty\n");
        return 1;
    }

    // Check if there is only one node in the queue
    if (nodeDel == queue -> back)
    {
        queue -> back = NULL;
    }

    queue -> head = nodeDel -> next;
    free(nodeDel);
    return 0;

}
