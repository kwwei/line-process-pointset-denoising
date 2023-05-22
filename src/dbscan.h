#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

using namespace std;

typedef struct dbPoint_
{
    float x, y, z;  // X, Y, Z position
    int clusterID;  // clustered ID
}dbPoint;

class DBSCAN {
public:    
    DBSCAN(unsigned int minPts, float eps, vector<dbPoint> points){
        m_minPoints = minPts;
        m_epsilon = eps;
        m_points = points;
        m_pointSize = points.size();
    }
    ~DBSCAN(){}

    int run();
    vector<int> calculateCluster(dbPoint point);
    int expandCluster(dbPoint point, int clusterID);
    inline double calculateDistance(const dbPoint& pointCore, const dbPoint& pointTarget);

    int getTotalPointSize() {return m_pointSize;}
    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilonSize() {return m_epsilon;}
    
public:
    vector<dbPoint> m_points;
    
private:    
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
};

#endif // DBSCAN_H
