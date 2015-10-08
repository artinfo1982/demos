package selenium;

import java.io.File;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedCondition;
import org.openqa.selenium.support.ui.WebDriverWait;

public class WebAutoTest4Chrome 
{
	public static void main(String[] args)
	{
		File file = new File("D:/opensource/chromedriver.exe");
		System.setProperty("webdriver.chrome.driver", file.getAbsolutePath()); 
		WebDriver driver = new ChromeDriver(); 
		driver.get("http://www.baidu.com");
		By by = By.id("kw");
		WebElement element = driver.findElement(by);
		element.sendKeys("华为");
		
		//等待页面跳转，10秒
		WebDriverWait wait = new WebDriverWait(driver, 10);	
		wait.until(new ExpectedCondition<WebElement>(){  
			@Override  
			public WebElement apply(WebDriver d) {  
				return d.findElement(By.id("content_left"));  
				}});
		
		driver.quit();
	}
}
