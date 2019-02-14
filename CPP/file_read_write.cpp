/*
* 使用C++ IO库操作文件示例
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string.h>

int read_file_ascii(const std::string &file_name, std::string &output)
{
    std::ifstream file(file_name, std::ios::in);
    if (!file.is_open())
    {
        std::cout << "file can not open, file name: " << file_name << std::endl;
        return 1;
    }
    std::string s;
    while (getline(file, s))
    {
        output.append(s);
        output.append("\n");
    }
    output.erase(output.end() -1);
    file.close();
    return 0;
}

int read_file_binary(const std::string &file_name, char *output, int buf_size)
{
    std::ifstream file(file_name, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "file can not open, file name: " << file_name << std::endl;
        return 1;
    }
    if (buf_size < 0)
    {
        std::cout << "buf_size  must greater than 0" << std::endl;
        return 1;
    }
    char *p = nullptr;
    while (file.read(p, buf_size))
        p += buf_size;
    file.close();
    return 0;
}

int write_file_ascii_append(const std::string &file_name, std::string &input)
{
    std::ofstream file(file_name, std::ios::out | std::ios::app);
    if (!file.is_open())
    {
        std::cout << "file can not open, file name: " << file_name << std::endl;
        return 1;
    }
    file << input;
    file.close();
    return 0;
}

int write_file_ascii_trunc(const std::string &file_name, std::string &input)
{
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (!file.is_open())
    {
        std::cout << "file can not open, file name: " << file_name << std::endl;
        return 1;
    }
    file << input;
    file.close();
    return 0;
}

int write_file_binary_append(const std::string &file_name, const char *input)
{
    std::ofstream file(file_name, std::ios::out | std::ios::app | std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "file can not open, file name: " << file_name << std::endl;
        return 1;
    }
    file << input;
    file.close();
    return 0;
}

int write_file_binary_trunc(const std::string &file_name, const char *input)
{
    std::ofstream file(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "file can not open, file name: " << file_name << std::endl;
        return 1;
    }
    file << input;
    file.close();
    return 0;
}

int main()
{
    std::string infile1("/home/cd/temp/a.txt");
    std::string infile2("/home/cd/temp/a.pb");
    std::string out;
    std::string outfile1("/home/cd/temp/tmp1.txt");
    std::string outfile2("/home/cd/temp/tmp2.txt");
    std::string outfile3("/home/cd/temp/tmp3.dat");
    std::string outfile4("/home/cd/temp/tmp4.dat");
    char *p1 = (char*)malloc(1024 * sizeof(char));
    memset(p1, 0x0, 1024);
    char p2[6] = {0x01, 0x02, 0x03, 0x04, 0x05};
    char p3[4] = {0x31, 0x32, 0x33};
    
    if (!read_file_ascii(infile1, out))
        std::cout << "read_file_ascii: " << std::endl << out << std::endl << std::endl;
    else
        std::cout << "ERROR, read_file_ascii" << std::endl;
    if (!read_file_binary(infile2, p1, 256))
    {
        std::cout << "read_file_binary:" << std::endl;
        int len = strlen(p1);
        for (int i=0; i<len; ++i)
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (0xff & p1[i]) << " ";
        std::cout << std::endl;
    }
    else
        std::cout << "ERROR, read_file_binary" << std::endl;
    for (int i=0; i<5; ++i)
    {
        if (write_file_ascii_append(outfile1, "aaa\n"))
            std::cout << "ERROR, write_file_ascii_append" << std::endl;
    }
    if (write_file_ascii_append(outfile2, "aaa\naaa"))
        std::cout << "ERROR, write_file_ascii_append" << std::endl;
    if (write_file_ascii_trunc(outfile2, "bbb\nbbb"))
        std::cout << "ERROR, write_file_ascii_trunc" << std::endl;
    if (write_file_binary_append(outfile3, p2))
        std::cout << "ERROR, write_file_binary_append" << std::endl;
    if (write_file_binary_append(outfile4, p2))
        std::cout << "ERROR, write_file_binary_append" << std::endl;
    if (write_file_binary_trunc(outfile4, p3))
        std::cout << "ERROR, write_file_binary_trunc" << std::endl;
    free(p1);
    p1 = nullptr;
    return 0;
}