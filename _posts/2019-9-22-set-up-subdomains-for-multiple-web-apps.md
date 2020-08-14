---
layout: post  
author: Qinkai Wu
title: Set up subdomains for multiple web Apps  
uniform_img_size: 650px  
pid: 2019092201  
tags: ['Web']
categories: journal
---

Suppose that you have registered a domain `mydomain.com`, and you are going to run multiple web applications either on a single or several devices, you may want to use subdomains for each of them like `app1.mydomain.com`, `app2.mydomain.com`, and so on.  

In my personal case, I have deployed JupyterLab, Rstudio, blog server and some other web apps on different machines located in my room and the lab. Many of them can not be directly accessed either because they do not have a public IP (e.g., [NAT](https://en.wikipedia.org/wiki/Network_address_translation) is used by many network providers) or are shielded under some firewall rules. The easiest and cheapest solution I found was to rent a VPS as a proxy while using tools like [**frp**](https://github.com/fatedier/frp), which is quite stable and I high recommend it.  

## Step-by-step How-to  
  
### 1. Add a new [glue record](https://www.cloudaccess.net/cloud-control-panel-ccp/157-dns-management/318-glue-records.html)
Log in to your account and open the settings panel provided by your domain registrar.   
Just fill in the two fields - nameserver and IP address.  
 
![add new record](/assets/img/post/{{ page.pid }}/gr.png)

### 2. Configure the Nginx  

Check the version of nginx:  
```shell
nginx -v
```

    nginx version: nginx/1.14.0 (Ubuntu)

Create a new folder to place your conf files  
```shell
mkdir /etc/nginx/site
```
Then write a simple conf file:  
```shell
vi /etc/nginx/site/test.conf
```

    server {
        listen 80;
        server_name test.mydomain.com;
        root /var/www/html;
        location / {
        }
    }


Add a simplest html file `index.html`:  
```shell
echo "hello" > /var/www/html/index.html
```

Make sure that the main nginx conf file looks like this:  
```shell
 cat /etc/nginx/nginx.conf
```

    user www-data;
    worker_processes auto;
    pid /run/nginx.pid;
    include /etc/nginx/modules-enabled/*.conf;

    events {
            worker_connections 768;
            # multi_accept on;
    }

    http {
            
            ##
            # Basic Settings
            ##
    
    ...


Now insert a line `include /etc/nginx/site/*.conf;` right after the `http{` to include the sub conf files in the `/etc/nginx/site` which we just created    


### 3. Open the Browser  

Go visit `test.mydomain.com` to see if it works!  

![test_hello](/assets/img/post/{{ page.pid }}/test_hello.png)  

Great! It's almost done!  


### 4. For apps running on different ports  

You can run many web apps on different ports of your VPS.  

Additionally, if you use `frp`, it will build a tunnel that connect two ports on your local machine and the VPS.  

Repeat step 1.  
This time in nginx conf file we don't have to specify `root `, but instead `proxy_pass`, because we are now use nginx as a reverse proxy rather than use it to host some static webpage files.   

```shell
vi /etc/nginx/site/new.conf
```

    server {
        listen 80;
        server_name rstudio.mydomain.com;
        location / {
                proxy_pass http://127.0.0.1:8787;
        }
    }

    server {
            listen 80;
            server_name jupyter.mydomain.com;
            location / {
                    proxy_pass http://127.0.0.1:8888;
    }


Again it works!  
![rstudio](/assets/img/post/{{ page.pid }}/r_st.png)



