package selenium;

import java.io.File;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.ie.InternetExplorerDriver;
import org.openqa.selenium.support.ui.ExpectedCondition;
import org.openqa.selenium.support.ui.WebDriverWait;

public class WebAutoTest4IE 
{
	public static void main(String[] args)
	{
		File file = new File("D:/opensource/IEDriverServer.exe");
		System.setProperty("webdriver.ie.driver", file.getAbsolutePath()); 
		WebDriver driver = new InternetExplorerDriver(); 
		driver.get("http://www.baidu.com");
		By by = By.id("kw");
		WebElement element = driver.findElement(by);
		element.sendKeys("华为");
		
		//等待页面跳转，10秒
		WebDriverWait wait = new WebDriverWait(driver, 60);	
		wait.until(new ExpectedCondition<WebElement>(){  
			@Override  
			public WebElement apply(WebDriver d) {  
				return d.findElement(By.id("content_left"));  
				}});
		
		driver.quit();
	}
}
