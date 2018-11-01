#ifndef CONFIG_H
#define CONFIG_H

class Config {
  
  public:
    Config();
    Config(int wrd_dim, int word_length, bool use_crf);
    void SetClassSize(int class_size);
    int  GetClassSize();
    ~Config();
  
  private:
    int wrd_dim;
    int chr_dim = 100;  // same as config.py
    int pos_dim = 6;    // same as config.py
    int etc_dim = 14;   // same as config.py
    int class_size;     // assigned after loading vocab
    int word_length;
    bool use_crf;
};

#endif
