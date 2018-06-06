# ntp的一些技巧

## windows server 2003作为ntp客户端向远端ntp服务器同步
```text
cmd下面执行：net time /setsntp:ntp服务器ip地址
重启time服务
```

## windows server 2008 r2作为ntp客户端向远端ntp服务器同步
```text
cmd下面执行：net time \\ntp服务器ip地址 /set
重启time服务
```
