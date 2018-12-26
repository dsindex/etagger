#ifndef CONFIG_H
#define CONFIG_H

class Config {
  
  public:
    Config();
    Config(int word_length, bool use_crf);
    void SetClassSize(int class_size) { this->class_size = class_size; }
    int  GetClassSize()  { return class_size; }
    int  GetWordLength() { return word_length; }
    bool GetUseCRF() { return use_crf; }
    ~Config();
  
  private:
    int class_size;     // assigned after loading vocab
    int word_length;
    bool use_crf;
};

#endif
