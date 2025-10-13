#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int key;
    struct Node* next;
} Node;

typedef struct Bucket {
    Node* head;
    pthread_mutex_t m;
} Bucket;

typedef struct {
    Bucket* buckets;
    int B;
} StripedList;

static int hash(int key, int B) { unsigned u = (unsigned)key; return (int)(u % (unsigned)B); }

void sl_init(StripedList* L, int B) {
    L->B = B;
    L->buckets = (Bucket*)calloc((size_t)B, sizeof(Bucket));
    for (int i=0;i<B;i++) pthread_mutex_init(&L->buckets[i].m, NULL);
}

void sl_destroy(StripedList* L) {
    for (int i=0;i<L->B;i++) {
        pthread_mutex_destroy(&L->buckets[i].m);
        Node* p=L->buckets[i].head;
        while(p){Node* q=p->next; free(p); p=q;}
    }
    free(L->buckets);
}

int sl_member(StripedList* L, int key) {
    int b = hash(key, L->B);
    pthread_mutex_lock(&L->buckets[b].m);
    Node* cur = L->buckets[b].head;
    while (cur && cur->key < key) cur = cur->next;
    int found = (cur && cur->key == key);
    pthread_mutex_unlock(&L->buckets[b].m);
    return found;
}

int sl_insert(StripedList* L, int key) {
    int b = hash(key, L->B);
    pthread_mutex_lock(&L->buckets[b].m);
    Node **pp = &L->buckets[b].head, *cur = L->buckets[b].head;
    while (cur && cur->key < key) { pp = &cur->next; cur = cur->next; }
    if (cur && cur->key == key) { pthread_mutex_unlock(&L->buckets[b].m); return 0; }
    Node* n = (Node*)malloc(sizeof(Node)); n->key=key; n->next=cur; *pp = n;
    pthread_mutex_unlock(&L->buckets[b].m);
    return 1;
}

int sl_delete(StripedList* L, int key) {
    int b = hash(key, L->B);
    pthread_mutex_lock(&L->buckets[b].m);
    Node **pp = &L->buckets[b].head, *cur = L->buckets[b].head;
    while (cur && cur->key < key) { pp = &cur->next; cur = cur->next; }
    if (!cur || cur->key != key) { pthread_mutex_unlock(&L->buckets[b].m); return 0; }
    *pp = cur->next; free(cur);
    pthread_mutex_unlock(&L->buckets[b].m);
    return 1;
}

/*** Demo con hilos: cada hilo hace operaciones aleatorias en la lista ***/
typedef struct { StripedList* L; int ops; unsigned seed; } ThArgs;

void* thread_ops(void* a_) {
    ThArgs* a = (ThArgs*)a_;
    for (int i=0;i<a->ops;i++) {
        int k = (int)(rand_r(&a->seed) % 100000);
        int op = (int)(rand_r(&a->seed) % 10);
        if (op<6) sl_member(a->L, k);       // 60% búsqueda
        else if (op<8) sl_insert(a->L, k);  // 20% inserción
        else sl_delete(a->L, k);            // 20% borrado
    }
    return NULL;
}

int main(int argc, char** argv){
    if (argc<5){fprintf(stderr,"uso: %s B hilos ops_por_hilo semillas...\n",argv[0]);return 1;}
    int B=atoi(argv[1]), T=atoi(argv[2]), OPS=atoi(argv[3]);
    StripedList L; sl_init(&L,B);

    pthread_t* th=(pthread_t*)malloc((size_t)T*sizeof(pthread_t));
    ThArgs* ta=(ThArgs*)malloc((size_t)T*sizeof(ThArgs));
    for (int i=0;i<T;i++){ ta[i]=(ThArgs){.L=&L,.ops=OPS,.seed=(unsigned)atoi(argv[4+i])};
        pthread_create(&th[i],NULL,thread_ops,&ta[i]); }
    for (int i=0;i<T;i++) pthread_join(th[i],NULL);

    sl_destroy(&L); free(ta); free(th);
    return 0;
}