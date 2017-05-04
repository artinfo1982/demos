/*
*
*
* ZooKeeper导入导出工具，从一个已有的zookeeper中将数据导出为文本文件，再将这个文本文件中的数据导入到一个新的zookeeper中。
* 导出：java -jar ZookeeperExportImport.jar -e -z 127.0.0.1:2181 -m 4000 -p "/Netrix" > a.txt
* 导入：java -jar ZookeeperExportImport.jar -i -z 127.0.0.1:2181 -m 4000 -f a.txt
*/


package test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;

import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.ZooDefs;
import org.apache.zookeeper.ZooKeeper;

public class ZookeeperExportImport {

	public static void main(String[] args) {

		if (args.length < 5) {
			System.out.println("Usage: -e/-i -z zkAddr -m sessionTimeout (-p path) (-f fileName)");
			System.exit(1);
		}

		String zkAddr = args[2];
		int sessionTimeout = Integer.valueOf(args[4]);
		Watcher watcher = new Watcher() {
			public void process(WatchedEvent event) {
			}
		};
		ZooKeeper zk = null;
		try {
			zk = new ZooKeeper(zkAddr, sessionTimeout, watcher);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

		if ("-e".equals(args[0])) {
			if (args.length != 7) {
				System.out.println("Usage: -e -z zkAddr -m sessionTimeout -p path");
				System.exit(1);
			}
			String path = args[6];
			zkExport(zk, path);
		} else if ("-i".equals(args[0])) {
			if (args.length != 7) {
				System.out.println("Usage: -i -z zkAddr -m sessionTimeout -f fileName");
				System.exit(1);
			}
			String fileName = args[6];
			zkImport(zk, fileName);
		}
	}

	private static void zkExport(ZooKeeper zk, String path) {
		List<String> child = null;
		try {
			child = zk.getChildren(path, false);
			if (child.size() == 0) {
				System.out.println(path + "=" + zk.getData(path, false, null));
			} else {
				for (String s : child) {
					zkExport(zk, path + "/" + s);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void zkImport(ZooKeeper zk, String fileName) {
		FileReader fr = null;
		BufferedReader br = null;
		String s = "";
		String path = "";
		String value = "";
		try {
			fr = new FileReader(fileName);
			br = new BufferedReader(fr);
			while ((s = br.readLine()) != null) {
				String[] kv = s.split("=");
				path = kv[0];
				String[] nodes = path.split("/");
				int length = nodes.length;
				int i;
				String pathTmp = "/";
				for (i = 1; i < length - 1; i++) {
					pathTmp = pathTmp + nodes[i];
					if (null == zk.exists(pathTmp, false)) {
						zk.create(pathTmp, new byte[0], ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
					}
					pathTmp += "/";
				}
				if (kv.length > 1) {
					value = kv[1].replaceAll("\r|\n", "");
					zk.create(path, value.getBytes(), ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
