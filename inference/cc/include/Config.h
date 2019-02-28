#ifndef CONFIG_H
#define CONFIG_H

class Config {
  
  public:
    Config();
    Config(int word_length);
    void SetClassSize(int class_size) { this->class_size = class_size; }
    int  GetClassSize()  { return class_size; }
    int  GetWordLength() { return word_length; }
    ~Config();
  
  private:
    int class_size;     // assigned after loading vocab
    int word_length;
};

#endif
