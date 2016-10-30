# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 20:21:50 2016

@author: Administrator
"""
'''
  python2.7实现
'''
import urllib2
import urllib
import bs4
import os
import re
'''
    开始爬虫
'''

#得到创建love文件的目录
path = os.getcwd()
path = os.path.join(path,"love")

#创建love文件
if not os.path.exists(path):
    os.mkdir(path)
        
url = "http://www.lovefou.com/dongtaitu/"
headers = {"User-Agent":"Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50"}
    

sum=48486-48300

def spider(first=48300,page_sum=sum):
    #储存Gif
    gif_dic = []
    for count in range(page_sum):
        req = urllib2.Request(url = url+str(first+count)+".html", headers = headers)
        try:
            response = urllib2.urlopen(req).read()
        
            soup = bs4.BeautifulSoup(response)
            img_content = soup.findAll('img')[:1]
            
            gif_src = img_content[0].get("src")
            gif_alt = img_content[0].get("alt")
            
            #判断两张Gif是否相同
            gif_src_dic = re.findall("/\w+/\w+\.gif",gif_src)[0][1:].split("/")
            found = gif_src_dic[0]
            rear = gif_src_dic[1][::-1][4:8]             
            gif_key = str(found)+str(rear)
            
            if not gif_key in gif_dic:
                gif_dic.append(gif_key)         
                filename = path+os.sep+gif_alt+".gif"
                print "正在下载图片%d"% count
                urllib.urlretrieve(gif_src,filename)
            else:
                continue
        except:
            print "链接不存在，跳过"
    
if __name__ == "__main__":
    spider()
