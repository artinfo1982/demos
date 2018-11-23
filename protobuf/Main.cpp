#include <iostream>
#include <fstream>
#include <string>
#include "test.pb.h"

int main(int argc, char *argv[])
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  tutorial::AddressBook address_book;
  
  tutorial::Person *person1 = address_book.add_person();
  tutorial::Person *person2 = address_book.add_person();
  
  tutorial::Person::PhoneNumber *person1_phone_number = person1->add_phone();
  tutorial::Person::PhoneNumber *person2_phone_number = person2->add_phone();
  
  person1->set_id(1);
  person1->set_name("a");
  person1->set_email("a@126.com");
  person1_phone_number->set_number("13913904062");
  person1_phone_number->set_type(tutorial::Person::MOBILE);
  
  person2->set_id(2);
  person2->set_name("b");
  person2->set_email("b@126.com");
  person2_phone_number->set_number("025-56620000");
  person2_phone_number->set_type(tutorial::Person::WORK);
  
  //序列化，存入pb文件
  std::fstream out("addressBook.pb", std::ios::out | std::ios::binary | std::ios::trunc);
  address_book.SerializeToOstream(&out);
  out.close();
  
  //反序列化，从pb文件中解析出信息并打印
  int i, j;
  for (i = 0; i < address_book.person_size(); ++i)
  {
    const tutorial::Person& person = address_book.person(i);
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Person ID: " << person.id() << std::endl;
    std::cout << "Person Name: " << person.name() << std::endl;
    if (person.has_email())
      std::cout << "E-mail address: " << person.email() << std::endl;
    for (j = 0; j < person.phone_size(); ++j)
    {
      const tutorial::Person::PhoneNumber& phone_number = person.phone(j);
      switch (phone_number.type())
      {
        case tutorial::Person::MOBILE:
          std::cout << "Mobile phone: ";
          break;
        case tutorial::Person::HOME:
          std::cout << "Home phone: ";
          break;
        case tutorial::Person::WORK:
          std::cout << "Work phone: ";
          break;
        default:
          break;
      }
      std::cout << phone_number().number() << std::endl;
    }
    std::cout << "----------------------------------" << std::endl << std::endl;
  }
  
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
