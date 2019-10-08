#include <iostream>
 
using namespace std;
 
int dp[30][30];
 
int main()
{
    int n = 10;
    dp[0][0] = 1;
    for(int i = 1; i<=n; ++i)
	{
        for(int j = i; j<=n; ++j)
            dp[i][j] = dp[i-1][j-1] + dp[i][j-i];
    }

    int a,b;
    while(~scanf("%d%d",&a,&b)){
        int ans = 0;
        for(int i = 1; i<=b; i++){
            ans += dp[i][a];
        }   
        cout << ans << endl;
    }
    return 0;
}
