#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ---------- Configuración fija ---------- */
enum { OPS_TOTALES = 100000, INIT_KEYS = 1000, MEMBER_PCT = 80, INSERT_PCT = 10, DELETE_PCT = 10 };

/* ---------- Estructuras ---------- */
typedef struct Node { int key; struct Node* next; } Node;

/* Lista única y spinlock global (busy waiting) */
static Node* head = NULL;
static atomic_flag spin = ATOMIC_FLAG_INIT;

/* ---------- Utilidades ---------- */
static inline void spin_lock(atomic_flag* f){
    while (atomic_flag_test_and_set_explicit(f, memory_order_acquire)) {
        /* busy-wait: quemamos CPU hasta que el lock se libere */
    }
}
static inline void spin_unlock(atomic_flag* f){
    atomic_flag_clear_explicit(f, memory_order_release);
}

static inline double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* RNG simple por hilo (xorshift32) */
typedef struct { unsigned x; } RNG;
static inline unsigned xr(RNG* r){ unsigned x=r->x; x^=x<<13; x^=x>>17; x^=x<<5; r->x=x?x:1u; return r->x; }

/* ---------- Operaciones (lista ordenada, protegidas por el spinlock global) ---------- */
static int ll_member(int key){
    spin_lock(&spin);
    Node* cur = head;
    while (cur && cur->key < key) cur = cur->next;
    int found = (cur && cur->key == key);
    spin_unlock(&spin);
    return found;
}
static int ll_insert(int key){
    spin_lock(&spin);
    Node **pp = &head, *cur = head;
    while (cur && cur->key < key) { pp=&cur->next; cur=cur->next; }
    if (cur && cur->key == key) { spin_unlock(&spin); return 0; }
    Node* n = (Node*)malloc(sizeof(Node)); n->key=key; n->next=cur; *pp=n;
    spin_unlock(&spin);
    return 1;
}
static int ll_delete(int key){
    spin_lock(&spin);
    Node **pp = &head, *cur = head;
    while (cur && cur->key < key) { pp=&cur->next; cur=cur->next; }
    if (!cur || cur->key != key) { spin_unlock(&spin); return 0; }
    *pp = cur->next; free(cur);
    spin_unlock(&spin);
    return 1;
}

/* ---------- Trabajo de cada hilo ---------- */
typedef struct { int ops; RNG rng; } ThArgs;

static void* worker(void* a_){
    ThArgs* a = (ThArgs*)a_;
    for (int i=0;i<a->ops;i++){
        int key  = (int)(xr(&a->rng) & 0x7fffffff);
        int pick = (int)(xr(&a->rng) % 100);
        if (pick < MEMBER_PCT)                 ll_member(key);
        else if (pick < MEMBER_PCT+INSERT_PCT) ll_insert(key);
        else                                    ll_delete(key);
    }
    return NULL;
}

/* ---------- Limpieza de la lista ---------- */
static void ll_free_all(void){
    Node* p=head; head=NULL;
    while(p){ Node* q=p->next; free(p); p=q; }
}

/* ---------- main ---------- */
int main(int argc, char** argv){
    if (argc<2){ fprintf(stderr,"uso: %s THREADS\n", argv[0]); return 1; }
    int T = atoi(argv[1]); if (T<=0){ fprintf(stderr,"THREADS inválido\n"); return 1; }

    /* carga inicial determinística */
    RNG r = {12345u};
    for (int i=0;i<INIT_KEYS;i++) ll_insert((int)(xr(&r)&0x7fffffff));

    /* hilos */
    pthread_t* th = (pthread_t*)malloc((size_t)T*sizeof *th);
    ThArgs*    ta = (ThArgs*)   malloc((size_t)T*sizeof *ta);

    int base = OPS_TOTALES / T, extra = OPS_TOTALES % T;
    double t0 = now_sec();
    for (int k=0;k<T;k++){
        ta[k].ops = base + (k < extra);
        ta[k].rng.x = 777u + (unsigned)k;
        pthread_create(&th[k], NULL, worker, &ta[k]);
    }
    for (int k=0;k<T;k++) pthread_join(th[k], NULL);
    double t1 = now_sec();

    printf("threads=%d  time=%.3f s  ops=%d  mix=%d/%d/%d\n",
           T, t1-t0, OPS_TOTALES, MEMBER_PCT, INSERT_PCT, DELETE_PCT);

    free(ta); free(th); ll_free_all();
    return 0;
}
