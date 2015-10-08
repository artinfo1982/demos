package net;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.handler.codec.string.StringDecoder;
import io.netty.handler.codec.string.StringEncoder;
import io.netty.util.CharsetUtil;

public class NettyTcpClient 
{
	public static String ip = "192.168.3.9";
	public static int port = 80;
	public static int threadNum = 32;
	public static String data = "GET / HTTP/1.1\nHost:192.168.3.9:80\nConnection:close\nContent-Length:0\n\n\r\n\r\n";
	
	public static Bootstrap bootstrap = getBootstrap();
	public static Channel channel = null;
	public static int success = 0;
	public static Object lock = new Object();
	
	public static final Bootstrap getBootstrap()
	{
		EventLoopGroup group = new NioEventLoopGroup();
		Bootstrap b = new Bootstrap();
		b.group(group).channel(NioSocketChannel.class);
		b.handler(new ChannelInitializer<Channel>() {
			@Override
			protected void initChannel(Channel ch) throws Exception 
			{
				ChannelPipeline pipeline = ch.pipeline();
				pipeline.addLast("decoder", new StringDecoder(CharsetUtil.UTF_8));
				pipeline.addLast("encoder", new StringEncoder(CharsetUtil.UTF_8));
				pipeline.addLast("handler", new TcpClientHandler());
			}
		});
		b.option(ChannelOption.TCP_NODELAY, true);
		b.option(ChannelOption.SO_REUSEADDR, true);
		return b;
	}
	
	public static void sendMsg(String host, int port, String msg)
	{
		try 
		{
			channel = bootstrap.connect(host, port).sync().channel();
		} catch (Exception e) 
		{
			System.err.println("get channel failed");
		}
		if(null != channel && channel.isActive())
		{
			try 
			{
				channel.writeAndFlush(msg).sync();
			} catch (Exception e) 
			{
				System.err.println("channel writeAndFlush failed");
			}
		}
		else
		{
			System.out.println("channel is null or channel is inactive");
		}
		channel.close();
	}
	
	public static class WorkingThread implements Runnable
	{
		public void run() 
		{
			for (;;)
			{
				NettyTcpClient.sendMsg(ip, port, data);
				synchronized (lock)
				{
					success ++;
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
}

class TcpClientHandler extends SimpleChannelInboundHandler<Object> 
{
	protected void channelRead0(ChannelHandlerContext ctx, Object msg) throws Exception 
	{
		//do something server response check
	}
}
