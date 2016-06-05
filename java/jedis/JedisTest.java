import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import redis.clients.jedis.HostAndPort;
import redis.clients.jedis.JedisCluster;

// redis cluster 采用的是分片技术，而不是热备份
// 本例子演示jedis如何操作redis集群
public class JedisTest {

	private static final String ip = "192.168.3.8";
	private static final int[] ports = { 6379, 6380, 6381, 6382, 6383, 6384 };

	public static void main(String[] args) throws Exception {

		Set<HostAndPort> nodes = new HashSet<HostAndPort>();
		for (int port : ports) {
			nodes.add(new HostAndPort(ip, port));
		}
		
		JedisCluster jc = new JedisCluster(nodes);
		
		jc.set("t1", "tank1");
		jc.set("t2", "tank2");
		System.out.println(jc.get("t1"));
		System.out.println(jc.get("t2"));
		jc.del("t1");
		jc.del("t2");
		
		byte[] passkey = "password".getBytes("UTF-8");
		byte[] password = "8H3gdlB09Gftw".getBytes("UTF-8");
		jc.set(passkey, password);
		String result = new String(jc.get(passkey), "UTF-8");
		System.out.println(result);
		jc.del(passkey);
		
		try {
			jc.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
