package net;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.string.StringDecoder;
import io.netty.handler.codec.string.StringEncoder;
import io.netty.util.CharsetUtil;

public class NettyTcpServer 
{
	private static final String ip = "192.168.3.9";
	private static final int port = 80;
	protected static final int threadGroupNum = Runtime.getRuntime().availableProcessors()*2;
	protected static final int workingThreadNm = 4;
	private static final EventLoopGroup bossGroup = new NioEventLoopGroup(threadGroupNum);
	private static final EventLoopGroup workerGroup = new NioEventLoopGroup(workingThreadNm);
	
	protected static void run() throws Exception 
	{
		ServerBootstrap b = new ServerBootstrap();
		b.group(bossGroup, workerGroup);
		b.channel(NioServerSocketChannel.class);
		b.childHandler(new ChannelInitializer<SocketChannel>() 
		{
			@Override
			public void initChannel(SocketChannel ch) throws Exception 
			{
				ChannelPipeline pipeline = ch.pipeline();
				pipeline.addLast("decoder", new StringDecoder(CharsetUtil.UTF_8));
				pipeline.addLast("encoder", new StringEncoder(CharsetUtil.UTF_8));
				pipeline.addLast(new TcpServerHandler());
			}
		});
		b.bind(ip, port).sync();
	}
	
	public static void main(String[] args) throws Exception
	{
		NettyTcpServer.run();
	}
}

class TcpServerHandler extends SimpleChannelInboundHandler<Object>
{
	@Override
	protected void channelRead0(ChannelHandlerContext ctx, Object msg)
			throws Exception 
	{
		System.out.println(msg);
		ctx.channel().writeAndFlush("aaa");
	}
	
	@Override
	public void exceptionCaught(ChannelHandlerContext ctx,
            Throwable cause) throws Exception 
	{
        ctx.close();
    }
}
