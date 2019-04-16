#ifndef RESULT_OBJ
#define RESULT_OBJ

#define MAX_WORD 64
#define MAX_POS  64
#define MAX_CHK  64
#define MAX_TAG  64
struct result_obj {
    char word[MAX_WORD];
    char pos[MAX_POS];
    char chk[MAX_CHK];
    char tag[MAX_TAG];
    char predict[MAX_TAG];
};
#endif
