---
layout: post  
author: Qinkai WU
title: Construct Private Cloud Storage with Seafile  
uniform_img_size: 800px  
id: 2019013001  
tags: ['Cloud Storage', 'Android Dev']
---

Ridiculously slow download speed, limited storage space, ... many people has long been sick of "Baidu Netdisk" and I'm no exception. So I was thinking why can't I build myself a cloud storage? It would be unrestrained, much faster and also ensures better privacy.

Then I found [`Seafile`](https://www.seafile.com/),
it's cross-platform, open-source and robust.

**Seafile website provides detailed and straightforward instructions so here I'm not gonna repeat too much of those. However through the practice, I did find some problems and here I will present my personal tips that would help.**

[Official instruction for deploying Seafile server](https://manual.seafile.com/deploy/)  
[Seafile server 安装教程中文版](https://manual-cn.seafile.com/deploy/)

***
## Basic Setup
  
First, download Seafile server at [https://www.seafile.com/download/](https://www.seafile.com/download/).
Alternatively, I decided to install the professional edition since it has more advanced features. To download the Pro version, you have to [create an account and sign up.](https://customer.seafile.com/)  
Anyway, the deploying procedures for them are almost the same.  

Then install all these prerequisites (Seafile does not support python3 currently).  

```shell
sudo apt update
sudo apt install deafault-jre
sudo apt install python2.7 libpython2.7 python-setuptools python-ldap python-urllib3 ffmpeg python-pip python-mysqldb python-memcache
sudo pip install pillow moviepy
```

I chose to use MySQL/MariaDB so next I mainly followed the steps [here](https://manual.seafile.com/deploy/using_mysql.html).  

Install mysql and set the password for root user.  

```shell
sudo apt install mariadb-server
sudo mysqladmin -u root password "my password"
```

After finishing the above, create an independent folder for seafile.  
I got a big hard drive and for convenience, I'll make a soft link.

```shell
mkdir "/media/`whoami`/MyDisk/Seafile"
cd ~ && ln -s "/media/`whoami`/MyDisk/Seafile" Seafile
```
Decompress the package and run setup script:

```shell
tar zxvf ~/Downloads/seafile-pro-server_*tar.gz -C ~/Seafile
cd ~/Seafile/seafile-pro-server-*
sudo ./setup-seafile-mysql.sh
```
You'll be asked to set some basic configurations, [here's my example]().

So far the server is ready for launch, start it using:  
(`seafile` is for files tranfer while `seahub` provides the web interface)  
```shell
sudo ./seafile.sh start
sudo ./seahub.sh start
```
Open `http://localhost:8000` in the browser and you'll see:  
![seahub](/img/blog/2019013001/seahub.png)
  
  
## Configure Nginx
  
  
Seafile and seahub takes up two ports.
Nginx can be used as a reverse proxy for them.

Create an nginx configure file:  
`vim ~/Seafile/seafile-server-latest/nginx.conf`  
My example of `nginx.conf` is [here](https://github.com/wu-2018/Collections/blob/master/Seafile_server_config/nginx.conf).

Next `sudo vim ~/Seafile/conf/seahub_settings.py`,  
Add a new line of code:  
`FILE_SERVER_ROOT = '/seafhttp'`  
(Note that I use `'/seafhttp'` rather than the full address like `'http://www.myseafile.com/seafhttp'` suggested by the official documentation. Then we can access seahub and download or upload files both through the local network IP like `http://192.168.1.105` or public network IP/domain.)
  
Restart  
```shell
sudo ./seafile.sh restart
sudo ./seahub.sh restart
```

 
## Expose the Local Server to the Internet
  
If the server is behind a NAT, tools like `ngrok` or `frp` can be used to expose it to the public internet so we can access the server remotely.  

[https://ngrok.com](https://ngrok.com) provides ngrok service and you can just download the client, it has serveral plans charging from 0 to $12/mo. But collectively considering the price and flexibility, I decided to build one since several months ago I already rent a VPS.  
Follow this to [build your own ngrok server and client](https://github.com/StudioEtrange/ngrok-build).  
  
Soon afterwards I found that `frp` might be a better choice, because we do not have to recompile it, the binary files are already released at [fatedier/frp](https://github.com/fatedier/frp), and they also do not ask for lengthy configurations.  

  
## Auto Start Seafile on System Boot

`sudo vim /etc/systemd/system/seafile.service` and add the following:


    [Unit]
    Description=Seafile
    # add mysql.service or postgresql.service depending on your database to the line below
    After=network.target mysql.service

    [Service]
    Type=oneshot
    ExecStart=/home/wqk/Seafile/seafile-server-latest/seafile.sh start
    ExecStop=/home/wqk/Seafile/seafile-server-latest/seafile.sh stop
    RemainAfterExit=yes
    User=root
    Group=root

    [Install]
    WantedBy=multi-user.target


Similar for `seahub` and `nginx`, just add a new file in `/etc/systemd/system/` repectively and change the key fields `ExecStart` and `ExecStop`.  

As for `frpc`, if I simply use `/home/wqk/tool/frpc -c /home/wqk/Seafile/frpc.ini` in its `ExecStart`, it would report error and exit upon bootup. That's because `frpc` tries to connect to its server even before the network is reachable. So I packed this command into another script `frpc.sh` and set the `ExecStart` as `/home/wqk/script/frpc.sh`:  
(`xxx.xx.xx.xxx` stands for the ip of the machine where `frps` is running)  
    
    #!/bin/bash     
    i=1
    sleep 20
    while (( $i<=100 ))
    do
    if ping -q -c 1 -W 1 xxx.xx.xx.xxx > /home/wqk/script/null; then
      echo "IPv4 is up"
      echo "starting frpc"
      /home/wqk/tool/frpc -c /home/wqk/Seafile/frpc.ini
      break
    else
      echo "IPv4 is down"
    fi
    sleep 2
    done

> #### [For reference, these config files can be found here](https://github.com/wu-2018/Collections/tree/master/Seafile_server_config)

  
Finally,  
```shell
sudo systemctl enable seafile.service
sudo systemctl enable seahub.service

sudo systemctl disable nginx.service
sudo systemctl enable nginx_seafile.service
sudo systemctl enable frpc.service

sudo systemctl daemon-reload
```
Now reboot the computer and all of them should automatically start.

  
## Revise the Seafile Android App
  
- **fix error when setting '/seafhttp'**  
As I just discussed above, after setting `FILE_SERVER_ROOT = '/seafhttp'` the web browser still works well. But in the Seafile Android app, I was logged in successfully but cannot download or upload files. After carefully checked the source code, I find the problems was at line 464 and 783 in `SeafConnection.java`, it won't get the full address in that case.  
![seadroid_1](/img/blog/2019013001/seadroid_1.png)  
(*left: FILE_SERVER_ROOT = 'http://172.17.15.11:8082';  right: FILE_SERVER_ROOT = '/seafhttp'*)  
[**See my changes**](https://github.com/haiwen/seadroid/compare/master...wu-2018:master). 
  

- **enable "double addresses"**  
Now I can login as if I have two accounts. I can access the files but it seems that seafile do not regard them as one single server. So it generate two folders in the phone and they're totally independent.   
Then I wondered if I could make some more modifications for this app so it can store a pair of server addresses and automatically switch to each other according to the network condition.   
I have to admit that I had zero basis in Java, but forced by the demand, I did a lot searching, tuned the code and failed many times, fortunately in the end my code passed through the compilation and the app works well on three Android phones I've tested. (Android version: 8 and 9; Brand: Xiaomi, Huawei and Meizu)  
The basic idea was saving a pair of addresses on login, then check whether the local IP is reachable or not.   
In `account/ui/AccountDetailActivity.java`, save the addresses pairs:  
```java
try {
    File file = new File(Environment.getExternalStorageDirectory(), ".seafile_server.txt");
    RandomAccessFile fos = new RandomAccessFile(file,"rw");
    String file_content = serverURL + '*' + server2URL + ";";
    fos.seek(file.length());
    fos.write(file_content.getBytes());
    fos.close();
    } catch (Exception e) {
        e.printStackTrace();
    }
```
Use `ping` to check the network in `SeafConnection.java`:  
```java
private Boolean pingIP(String HTTPaddress) {
        String[] IPandPort_  = HTTPaddress.split("/");
        String[] IP_  = IPandPort_[2].split(":");

        try {
            Process p = Runtime.getRuntime().exec("ping -c 2 -w 4 "+IP_[0]);
            int status = p.waitFor();
            if (status == 0) {
                return true;
            } else {
                return false;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
```
See the change of the logging interface:  
![seadroid_2](/img/blog/2019013001/seadroid_2.png)  
(*left: official version;  right: revised version*)  
> #### [**See more changes in the source code**](https://github.com/haiwen/seadroid/compare/master...wu-2018:PairAddr).  
  
After being tortured by java, finally I had a deep understanding for "Life is short, I use python".
