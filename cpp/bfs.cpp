#include<bits/stdc++.h>

using namespace std;

typedef pair<int,int> ii;

const int N = 1e7+10;
const int K = 1e5+10;
int n, m, k;

vector<int> adj[N];
int src[K];
vector<ii> shortest_path[K];


int dist[N];
void bfs(int isrc){
    memset(dist,-1,sizeof(dist));
    queue<int> q;

    q.push(src[isrc]);
    dist[src[isrc]] = 0;

    while(q.size()){

        int u = q.front(); q.pop();
//        cout<<u<<endl;
        shortest_path[isrc].push_back(ii(u,dist[u]));

        for(int v : adj[u]){
//            cout<<" "<<v<<endl;
//            exit(1);
            if(dist[v]!=-1) continue;
            q.push(v);
            dist[v] = dist[u]+1;
        }
    }

}

int main(){

    scanf("%d%d%d",&n,&m,&k);
    for(int i = 0 ; i < k ; i++){
        scanf("%d",src+i);
    }
    for(int i = 0 ; i < m ; i ++){
        int a,b;
        scanf("%d%d",&a,&b);
        // assuming that we get each edge twice (once in each direction)
        adj[a].push_back(b);
    }

    cout<<"{"<<endl;
    for(int i = 0 ;i < k ; i ++){
        bfs(i);
        cout<<src[i]<<": {";
        for(ii p:shortest_path[i]){
            printf("%d: %d%s",p.first, p.second, (p == shortest_path[i].back())?"":", ");
        }
        cout<<"}"<<((i+1==k)? "" : ", ");
    }
    cout<<"}"<<endl;




    return 0;
}

/**

6 8 3
0 2 4
0 2
0 4
0 5
1 4
1 5
2 3
2 4
4 5

**/
