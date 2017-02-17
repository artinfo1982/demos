#! /usr/bin/expect

set ip [lindex $argv 0]
set passwd [lindex $argv 1]

spawn ssh root@${ip} "date"
expect {
  "(yes/no)?" {
    send "yes\r"
    expect "assword:"
    send "${passwd}\r"
  }
  "assword:" {send "${passwd}\r"}
  "* to host" {exit 0}
}
expect eof
