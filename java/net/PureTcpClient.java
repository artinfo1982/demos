package net;

import java.io.BufferedInputStream;
import java.io.OutputStream;
import java.net.Socket;

public class PureTcpClient
{
	public static String ip = "192.168.3.9";
	public static int port = 2000;
	public static int threadNum = 32;
	public static byte [] data = new String(
			"GET / HTTP/1.1\nHost:" + ip + ":" + port + 
			"\nConnection:close\nContent-Length:0\n\n\r\n\r\n").getBytes();
	public static int success = 0;
	public static Object lock = new Object();
	
	public static void main(String[] args)
	{
		int i;
		PrintThread p = new PrintThread();
		Thread t0 =new Thread(p);
		WorkingThread w = new WorkingThread();
		Thread [] t1 = new Thread[threadNum];
		for (i=0; i<threadNum; i++)
		{
			t1[i] =new Thread(w);
		}
		t0.start();
		for (i=0; i<threadNum; i++)
		{
			t1[i].start();
		}
		try 
		{
			t0.join();
			for (i=0; i<threadNum; i++)
			{
				t1[i].join();
			}
		} catch (Exception e) {}
	}
	
	public static class WorkingThread implements Runnable
	{
		@SuppressWarnings("unused")
		public void run() 
		{
			BufferedInputStream  in = null;
			OutputStream out = null;
			Socket client = null;
			byte [] buf = new byte[1024];
			int readCount = 0;
			
			for (;;)
			{				
				try
				{
					client = new Socket(ip, port);
					in = new BufferedInputStream(client.getInputStream());
					out = client.getOutputStream();
					out.write(data);
					client.shutdownOutput();
					while ((readCount = in.read(buf)) != -1)
					{
						//do something body check
					}
					synchronized (lock)
					{
						success++;
					}							
				}
				catch (Exception e){}
				finally
				{
					if (null != in)
					{
						try 
						{
							in.close();
						} catch (Exception e) {}
					}
					if (null != out)
					{
						try 
						{
							out.close();
						} catch (Exception e) {}
					}
					if (null != client)
					{
						try 
						{
							client.close();
						} catch (Exception e) {}
					}
				}
			}
		}
	}
	
	public static class PrintThread implements Runnable
	{
		public static int t1;
		public static int t2;
		public void run() 
		{
			for (;;)
			{
				t1 = success;
				try 
				{
					Thread.sleep(4000);
				} catch (InterruptedException e) {}
				t2 = success;
				System.out.println("[" + ((t2 - t1) >> 2) + "]/s");
			}
		}	
	}
}
