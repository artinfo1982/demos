package test;

import java.util.ArrayList;
import java.util.List;

import org.bson.Document;

import com.mongodb.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.InsertOneModel;
import com.mongodb.client.model.WriteModel;

public class MongoDbBulkInsert implements Runnable {

	private static String mongoDbIp = "";
	private static int mongoDbPort = 0;
	private static String mongoDbName = "";
	private static String mongoDbCollection = "";
	private static long itemNum = 0L;
	private static int fileSize = 0;
	private static int threadNum = 0;

	@Override
	public void run() {
		byte file[] = new byte[fileSize];
		int i;
		@SuppressWarnings("resource")
		MongoClient mc = new MongoClient(mongoDbIp, mongoDbPort);
		MongoDatabase mdb = mc.getDatabase(mongoDbName);
		MongoCollection<Document> col = mdb.getCollection(mongoDbCollection);
		for (;;) {
			List<WriteModel<Document>> docs = new ArrayList<WriteModel<Document>>();
			Document doc = new Document();
			for (i = 0; i < itemNum; i++) {
				doc.append("a", file);
			}
			docs.add(new InsertOneModel<Document>(doc));
			col.bulkWrite(docs);
		}
	}

	public static void main(String[] args) {
		mongoDbIp = args[0];
		mongoDbPort = Integer.valueOf(args[1]);
		mongoDbName = args[2];
		mongoDbCollection = args[3];
		itemNum = Long.valueOf(args[4]);
		fileSize = Integer.valueOf(args[5]);
		threadNum = Integer.valueOf(args[6]);

		Thread[] threadGroup = new Thread[threadNum];
		for (int i = 0; i < threadNum; i++) {
			threadGroup[i] = new Thread(new MongoDbBulkInsert());
			threadGroup[i].start();
		}
		@SuppressWarnings("resource")
		MongoClient mc = new MongoClient(mongoDbIp, mongoDbPort);
		MongoDatabase mdb = mc.getDatabase(mongoDbName);
		MongoCollection<Document> col = mdb.getCollection(mongoDbCollection);
		long beg, end, diff;
		int caps;
		for (;;) {
			beg = col.count();
			try {
				Thread.sleep(4000);
			} catch (InterruptedException e) {
			}
			end = col.count();
			diff = (end - beg) * itemNum;
			caps = (int) (diff >> 2);
			System.out.println(caps);
		}
	}
}
