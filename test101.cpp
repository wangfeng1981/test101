#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include "stdio.h"
#include <string>
#include "math.h"

#define CIM_THRESHOLD 50

using namespace std ;
using namespace cv ;

struct wfKeypoint{
	int flag ;
	int irow ;
	int icol ;
	int rows ;
	int cols ;
	double cim ;
	double mag ;
	double ori ;
	double desc[128] ;
} ;

void writeKpArrayToFile( const char* path , wfKeypoint* kpArr , int nkp ){
	FILE* pf = fopen( path , "w" ) ;
	fprintf(pf,"rows,cols\n") ;
	fprintf(pf, "%d,%d\n", 128 , nkp );
	fprintf(pf, "data\n" );
	for(int i = 0 ; i<128 ; ++ i ){
		for(int j = 0 ; j<nkp ; ++ j ){
			fprintf(pf,"%8.5f " , kpArr[j].desc[i] ) ;
		}
		fprintf(pf, "\n");
	}
	
	fclose(pf) ;
	pf = NULL ;
}

void computeDist( vector< DMatch > v , double* pmin , double* pmax ){

	double max_dist = v[0].distance ; double min_dist = v[0].distance ;
	for( int i = 0; i < v.size() ; i++ )
	{ 
		double dist = v[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}
	*pmin = min_dist ;
	*pmax = max_dist ;
}

void extractGoodMatches( vector<DMatch> allmatch , 
							vector<DMatch>& goodmatch ,
							double* mean_val ,
							double* std_val  ){
	double dist0 , dist1 ;
	double meangooddist , stdgooddist ;
	computeDist( allmatch , &dist0 , &dist1 ) ;
	for(int i = 0 ; i<allmatch.size() ; ++ i ){
		if( allmatch[i].distance <= fmax(2*dist0, 0.02) ) {
			goodmatch.push_back( allmatch[i]);
			meangooddist += allmatch[i].distance ;
		}
	}
	meangooddist = meangooddist / goodmatch.size() ;
	for(int i = 0 ; i<goodmatch.size() ; ++ i ){
		double dd = goodmatch[i].distance - meangooddist ;
		stdgooddist += dd*dd ;
	}
	stdgooddist = sqrt( stdgooddist / fmax( 1, goodmatch.size() -1 ) ) ;

	*mean_val = meangooddist ;
	*std_val = stdgooddist ;
}

void readKeypointsAndDesc( char* ymlpath , std::vector<KeyPoint>& ref_keypoints , Mat& ref_desc ){
	FileStorage fs ;
	fs.open(  ymlpath , FileStorage::READ);
    if(fs.isOpened()){
            read(fs["desc"], ref_keypoints ); 
            read(fs["kps"], ref_desc);   
    }   
	fs.release();
}


void print2dMatFloat( Mat mat  ){
	printf("\n Mat: \n");
	for( int irow = 0 ; irow < mat.rows ; ++ irow ){
		for(int icol = 0 ; icol < mat.cols ; ++ icol ){
			printf("%8.4f ", mat.at<float>(irow,icol) );
		}
		printf( " \n " ) ;
	}
}


void print2dMatFloat2Int( Mat mat  ){
	printf("\n Mat: \n");
	for( int irow = 0 ; irow < mat.rows ; ++ irow ){
		for(int icol = 0 ; icol < mat.cols ; ++ icol ){
			printf("%3d ",(int) mat.at<double>(irow,icol) );
		}
		printf( " \n " ) ;
	}
}

void print2dMatDouble( Mat mat  ){
	printf("\n Mat: \n");
	for( int irow = 0 ; irow < mat.rows ; ++ irow ){
		for(int icol = 0 ; icol < mat.cols ; ++ icol ){
			printf("%8.4f ", mat.at<double>(irow,icol) );
		}
		printf( " \n " ) ;
	}
}

//uchar , float , double
void print2dMatToFile( const char* filepath , Mat mat , const char* typestr ){
	int type = 0 ;//uchar
	if( strcmp(typestr,"float") == 0) type=1 ;
	if( strcmp(typestr,"double") ==0 ) type = 2 ;
	FILE* pf = fopen( filepath , "w") ;
	if( pf ){
		fprintf( pf, "rows,cols\n") ;
		fprintf( pf, "%d , %d \n", mat.rows , mat.cols );
		fprintf( pf , "data\n");
		for( int irow = 0 ; irow < mat.rows ; ++irow ){
			for(int icol = 0 ; icol < mat.cols ; ++ icol ){
				if( type==1 ){
					fprintf( pf , "%8.4f ", mat.at<float>(irow,icol) );
				}else if( type==2 ){
					fprintf( pf , "%8.4f ", mat.at<double>(irow,icol) );
				}else {
					fprintf( pf , "%d ", mat.at<uchar>(irow,icol) );
				}
				
			}
			fprintf(pf,"\n") ;
		}
	}
	fclose(pf) ;
	pf = NULL ;
}

Mat read2dMatFromFile( const char* filepath  ){
	FILE* pf = fopen( filepath , "r") ;
	Mat mat1 ;
	if( pf ){
		int cols , rows ;
		char buff[100] ;
		fscanf( pf , "%s" , buff) ;
		fscanf( pf , "%d , %d" , &rows , &cols ) ;
		printf("rows , cols : %d , %d \n" , rows , cols) ;

		fscanf( pf , "%s" , buff) ;
		Mat mat(rows , cols , CV_64F ) ;
		for( int irow = 0 ; irow < mat.rows ; ++irow ){
			for(int icol = 0 ; icol < mat.cols ; ++ icol ){
				float val = 0 ;
				fscanf( pf , "%f " , &val ) ;
				mat.at<double>(irow,icol)  =  val ;
			}
		}
		mat1 = mat ;
	}
	fclose(pf) ;
	pf = NULL ;
	return mat1 ;
}

Mat convert01( Mat mat ){
	Mat mat1(mat.rows, mat.cols, CV_64F );
	for(int i = 0 ; i<mat.rows ; ++ i ){
		for(int j = 0 ; j<mat.cols ; ++ j ){
			mat1.at<double>(i,j) = mat.at<uchar>(i,j) / 255.0 ;
		}
	}
	return mat1 ;
}

Mat convertdouble( Mat mat ){
	Mat mat1(mat.rows, mat.cols, CV_64F );
	for(int i = 0 ; i<mat.rows ; ++ i ){
		for(int j = 0 ; j<mat.cols ; ++ j ){
			mat1.at<double>(i,j) = mat.at<uchar>(i,j) * 1.0 ;
		}
	}
	return mat1 ;
}

Mat convert0255( Mat mat ){
	Mat mat1(mat.rows, mat.cols, CV_8U );
	for(int i = 0 ; i<mat.rows ; ++ i ){
		for(int j = 0 ; j<mat.cols ; ++ j ){
			mat1.at<uchar>(i,j) = (int)(mat.at<double>(i,j) * 255.0) ;
		}
	}
	return mat1 ;
}

void showit( const char* windowname , Mat mat ){
	namedWindow( windowname , WINDOW_AUTOSIZE ) ;
	imshow( windowname , mat ) ;
}

void printDebugData8U( const char* label, Mat mat , int irow , int icol ){
	printf("%s\n" , label ) ;
	for( int i =irow-1 ; i<=irow+1 ; ++ i ){
		for(int j = icol-1 ; j<=icol+1 ; ++ j ){
			printf( "%3d " , mat.at<uchar>(i,j) ) ;
		}
		printf("\n") ;
	}
}

void printDebugDataDouble( const char* label, Mat mat , int irow , int icol ){
	printf("%s\n" , label ) ;
	for( int i =irow-1 ; i<=irow+1 ; ++ i ){
		for(int j = icol-1 ; j<=icol+1 ; ++ j ){
			printf( "%6.2f " , mat.at<double>(i,j) ) ;
		}
		printf("\n") ;
	}
}




void printPixelArounded( const char* label, 
	Mat& mat , int irow , int icol , int rad ){
	printf("%s\n" , label ) ;
	for( int i = irow - rad ; i<=irow+rad ; ++ i ){
		for(int j = icol-rad ; j<=icol+rad ; ++ j ){
			if( i>=0 && i< mat.rows && j>=0 && j<mat.cols ){
				if( i==irow && j==icol ){
					printf("[%6.2f], " , mat.at<double>(i,j) ) ;
				}else{
					printf(" %6.2f , " , mat.at<double>(i,j) ) ;	
				}
			}else{
				printf(" %6.2f , " , -99.99 ) ;
			}
		}
		printf("\n") ;
	}

}



void computePixelMagOri( Mat mat , int irow , int icol ,
	double& mag ,int& ori ){
	if( irow <1 || icol<1 || irow>mat.rows-2 || icol>mat.cols-2 ){
		mag = 0 ;
		ori = 0 ;
	}else{
		double p10 = 1.0*mat.at<double>(irow+1,icol) ;
		double pn0 = 1.0*mat.at<double>(irow-1,icol) ;
		double p01 = 1.0*mat.at<double>(irow,icol+1) ;
		double p0n = 1.0*mat.at<double>(irow,icol-1) ;

		mag = sqrt( (p10-pn0)*(p10-pn0) 
			+ (p01-p0n)*(p01-p0n) ) ;
		double deg = atanf( std::abs(p10-pn0)/std::abs(p01-p0n) ) * 180.0 / 3.1415926 ;

		if( p10-pn0 > 0 && p01-p0n < 0 ){
			deg += 90 ;
		}else if( p10-pn0 < 0 && p01-p0n < 0 ){
			deg += 180 ;
		}else if( p10-pn0 < 0 && p01-p0n > 0 ){
			deg += 270 ;
		}

		ori = 0 ;
		for(int i = 0 ; i<36 ; ++ i ){
			if( deg < i*10 ){
				ori = i ;
				break; 
			}
		}
		/*
		printf("irow icol %d %d \n" , irow , icol ) ;
		printf("irow+1,icol uchar : %f \n" , mat.at<double>(irow+1,icol) ) ;
		printf(" %8.4f %8.4f %8.4f %8.4f \n" , p10,pn0,p01,p0n ) ;
		printf(" mag deg %8.4f %8.4f \n" , mag , deg ) ;
		printf(" ori index %d \n" , ori ) ;*/
		
 
	}
	
}

void computeKeyPointMagOri(Mat mat , int irow , int icol , double& mag , int& ori ){

	int rad = 3 ;
	double mags[36] ;
	for(int i = 0 ; i<36 ; ++ i ) mags[i] = 0 ;
	for(int i = irow - rad ; i<= irow+rad ; ++ i ){
		for(int j = icol-rad ; j<=icol+rad ; ++ j ){
			if( i<0 || j<0 || i>mat.rows-1 || j>mat.cols-1 ){
				continue ;
			}else if( i!=irow || j!=icol ) {
				double mag1 = 0 ;
				int ori1 = 0 ;
				computePixelMagOri(mat,i,j,mag1,ori1) ;
				mags[ori1] += mag1 ;
				//printf("%d " , ori1 ) ;
			}
		}
	}
	
	mag = 0 ;
	ori = 0 ;
	for (int i = 0; i < 36 ; ++i)
	{
		if( mags[i] > mag ){
			mag = mags[i] ;
			ori = i ;
		}
	}
	printf("\n") ;
}
 
void setMatValue8U( Mat mat , int irow , int icol , uchar val ){
	if( irow < 0 || icol < 0 || irow > mat.rows-1 || icol > mat.cols-1 ){
		return ;
	}else{
		mat.at<uchar>(irow,icol) = val ;
	}
}


void computePixelMagOriForDesc( Mat matd , int irow , int icol ,
	double& mag ,int& ori8 ){
	if( irow <1 || icol<1 || irow>matd.rows-2 || icol>matd.cols-2 ){
		mag = 0 ;
		ori8 = 0 ;
	}else{
		double p10 = 1.0*matd.at<double>(irow+1,icol) ;
		double pn0 = 1.0*matd.at<double>(irow-1,icol) ;
		double p01 = 1.0*matd.at<double>(irow,icol+1) ;
		double p0n = 1.0*matd.at<double>(irow,icol-1) ;

		mag = sqrt( (p10-pn0)*(p10-pn0) 
			+ (p01-p0n)*(p01-p0n) ) ;
		double deg = atanf( std::abs(p10-pn0)/std::abs(p01-p0n) ) * 180.0 / 3.1415926 ;

		if( p10-pn0 > 0 && p01-p0n < 0 ){
			deg += 90 ;
		}else if( p10-pn0 < 0 && p01-p0n < 0 ){
			deg += 180 ;
		}else if( p10-pn0 < 0 && p01-p0n > 0 ){
			deg += 270 ;
		}

		ori8 = 0 ;
		for(int i = 0 ; i<8 ; ++ i ){
			if( deg < i*45 ){
				ori8 = i ;
				break; 
			}
		}
		/*
		printf("irow icol %d %d \n" , irow , icol ) ;
		printf("irow+1,icol uchar : %f \n" , matd.at<double>(irow+1,icol) ) ;
		printf(" %8.4f %8.4f %8.4f %8.4f \n" , p10,pn0,p01,p0n ) ;
		printf(" mag deg %8.4f %8.4f \n" , mag , deg ) ;
		printf(" ori8 index %d \n" , ori8 ) ;*/
	}	
}

void computeKpDescForOneWindow( Mat matd , int irow0 , int icol0 , double* window8 ){
	for(int i = 0 ; i<8 ; ++ i ) window8[i] = 0 ;
	double weighted[] = {
		0.015 , 0.047 , 0.047 , 0.015
	 , 0.047 , 0.142 , 0.142 , 0.047
	 , 0.047 , 0.142 , 0.142 , 0.047
	 , 0.015 , 0.047 , 0.047 , 0.015
	} ;
	int iw = 0 ;
	for( int i = irow0 ; i< irow0+4 ; ++ i ){
		for(int j = icol0 ; j< icol0+4 ; ++ j ){

			if( i<0 || j<0 || i>matd.rows-1 || j>matd.cols-1 ){

			}else{
				double mag = 0 ;
				int ori8 = 0 ;
				computePixelMagOriForDesc( matd , i,j , mag , ori8 ) ;
				window8[ori8] += mag * weighted[iw] ;
			}
			iw++ ;
		}
	}
}

void normalizeVector( double* array0 ,double* resArray, int size ){
	double* tarray = new double[size] ;
	double mo = 0 ;
	for(int i = 0 ; i<size ; ++ i ){
		mo += array0[i]*array0[i] ;
	}
	mo = sqrtf( mo ) ;
	for(int i = 0 ; i<size ; ++ i ){
		tarray[i] = array0[i]/mo ;
	}
	for(int i =0 ; i<size;  ++ i ){
		resArray[i] = tarray[i] ;
	}
	delete[] tarray ;
}

void computeKpDesc( Mat matd , int irow , int icol ,double* desc128 ) {
	for(int i = 0 ; i<128 ; ++ i ) desc128[i] = 0 ;
	int i8 = 0 ;
	for( int i = irow - 7; i<= irow+8 ; i=i+4 ){
		for(int j = icol-7 ; j<=icol+8 ; j=j+4 ){
			if( i<0 || j<0 || i>matd.rows-1 || j>matd.cols-1 ){
				
			}else{
				double w8[8] ;
				computeKpDescForOneWindow(matd , i,j, w8 ) ;
				for(int ii = 0; ii<8 ; ++ ii ){
					desc128[i8+ii] = w8[ii] ;
				}
			}
			i8 += 8 ;
		}
	}
	normalizeVector( desc128 , desc128 , 128 ) ;
}

/*
kpmat0.at<uchar>(i-1,j-1) = 255 ;
						kpmat0.at<uchar>(i-1,j) = 255 ;
						kpmat0.at<uchar>(i-1,j+1) = 255 ;
						kpmat0.at<uchar>(i,j-1) = 255 ;
						kpmat0.at<uchar>(i,j) = 255 ;
						kpmat0.at<uchar>(i,j+1) = 255 ;
						kpmat0.at<uchar>(i+1,j-1) = 255 ;
						kpmat0.at<uchar>(i+1,j) = 255 ;
						kpmat0.at<uchar>(i+1,j+1) = 255 ;
*/

void wfsiftFindKeypoints(Mat grayd, Mat dogup, Mat dogmid , Mat dogdn , 
	std::vector<wfKeypoint>& wfkpVector ){

	for(int i = 1 ; i<dogmid.rows-1 ; ++ i ){
		for( int j = 1; j<dogmid.cols-1 ; ++j ){
			int nbig = 0 ;
			int nsmall = 0 ;
			double cur = dogmid.at<double>(i,j) ;
			double curd = grayd.at<double>(i,j) ;

			for( int i1=i-1 ; i1<=i+1 ; ++ i1 ){
				for(int j1=j-1 ; j1<=j+1 ; ++ j1 ){

					if( cur >= dogup.at<double>(i1,j1) ){
						nbig++ ;
					}
					if( cur <= dogup.at<double>(i1,j1) ){
						nsmall++ ;
					}
					if( cur >= dogdn.at<double>(i1,j1) ){
						nbig++ ;
					}
					if( cur <= dogdn.at<double>(i1,j1) ){
						nsmall++ ;
					}
					if( i == i1 && j==j1  ){

					}else{
						if( cur >= dogmid.at<double>(i1,j1) ){
							nbig++ ;
						}
						if( cur <= dogmid.at<double>(i1,j1) ){
							nsmall++ ;
						}
					}
				}
			}
			if( nbig == 26 || nsmall==26 ){
				//kick out some bad kp!
			
				if( curd > 8 ){

					double dxx = dogmid.at<double>(i,j-1)
						+ dogmid.at<double>(i,j+1)
						- 2*cur ;
					double dyy = dogmid.at<double>(i-1,j)
						+ dogmid.at<double>(i+1,j)
						- 2*cur ;
					double  dxy = (dogmid.at<double>(i-1,j-1)
						+ dogmid.at<double>(i+1,j+1)
						- dogmid.at<double>(i-1,j+1)
						- dogmid.at<double>(i+1,j-1))/4.0 ;

					/*
					printf("%8.2f %8.2f %8.2f \n%8.2f %8.2f %8.2f \n%8.2f %8.2f %8.2f \n ",
						dogmid.at<double>(i-1,j-1) , 
						dogmid.at<double>(i  ,j-1) , 
						dogmid.at<double>(i+1,j-1) , 
						dogmid.at<double>(i-1,j ) , 
						dogmid.at<double>(i  ,j ) , 
						dogmid.at<double>(i+1,j ) , 
						dogmid.at<double>(i-1,j+1) , 
						dogmid.at<double>(i  ,j+1) , 
						dogmid.at<double>(i+1,j+1) 
						) ;

					printf("dxx dyy dxy %8.2f %8.2f %8.2f\n",dxx,dyy,dxy) ;
					*/
					double trH = dxx + dyy;
					double detH = dxx*dyy - dxy*dxy;

					double curvature_ratio = trH*trH/detH;

					
					if(detH>0 && curvature_ratio > 10 )
					{
						double mag = 0 ;
						int ori = 0 ;
						computeKeyPointMagOri( grayd , i,j,mag,ori ) ;
						wfKeypoint thekp ;
						thekp.rows = dogmid.rows ;
						thekp.cols = dogmid.cols ;
						thekp.irow = i ;
						thekp.icol = j ;
						thekp.mag = mag ;
						thekp.ori = ori ;
						computeKpDesc(grayd , i,j ,thekp.desc ) ;
						wfkpVector.push_back(thekp) ;
					}
				}
			}
		}
	}
}


void wfsiftComputeOctave( Mat grayd , double* sigArr,
	std::vector<wfKeypoint>& kpVector) {
	printf("oct %d \n" , grayd.rows ) ;

	Mat blur1 , blur2 , blur3 , blur4 ;
	GaussianBlur( grayd, blur1 , Size(0,0) , sigArr[0] ) ;
	GaussianBlur( blur1, blur2 , Size(0,0) , sigArr[1]  ) ;
	GaussianBlur( blur2, blur3 , Size(0,0) , sigArr[2]  ) ;
	GaussianBlur( blur3, blur4 , Size(0,0) , sigArr[3]  ) ;

	Mat dog0 ; subtract( grayd , blur1 ,dog0) ;
	Mat dog1 ; subtract( blur1 , blur2, dog1) ;
	Mat dog2 ; subtract( blur2 , blur3, dog2) ;
	Mat dog3 ; subtract( blur3 , blur4, dog3) ;

	int tx , ty ;
	tx = (int) ( (104.7/256)* grayd.cols ) ;
	ty = (int) ( (151.4/256)* grayd.rows ) ;
	printPixelArounded( "blur0" , grayd , ty,tx , 5 ) ;
	printPixelArounded( "blur1" , blur1 , ty,tx , 5 ) ;
	printPixelArounded( "blur2" , blur2 , ty,tx , 5 ) ;
	printPixelArounded( "blur3" , blur3 , ty,tx , 5 ) ;
	printPixelArounded( "blur4" , blur4 , ty,tx , 5 ) ;

	printPixelArounded( "dog0" , dog0 , ty,tx , 5 ) ;
	printPixelArounded( "dog1" , dog1 , ty,tx , 5 ) ;
	printPixelArounded( "dog2" , dog2 , ty,tx , 5 ) ;
	printPixelArounded( "dog3" , dog3 , ty,tx , 5 ) ;


	wfsiftFindKeypoints( grayd , dog0 , dog1 , dog2 , kpVector ) ;
	printf("Find kps 1: %d \n" , (int)kpVector.size() ) ;
	wfsiftFindKeypoints( grayd , dog1 , dog2 , dog3 , kpVector ) ;
	printf("Find kps 2: %d \n" , (int)kpVector.size() ) ;

}

void wfsiftRemoveRepeatKeyPoints( std::vector<wfKeypoint>& vector1 ){
	std::vector<wfKeypoint> vector2 ;

	int n1 = vector1.size() ;
	for(int i = 0 ; i<n1 ; ++ i ) {

		wfKeypoint kp1 = vector1[i] ;
		if( kp1.flag == -1 ) {
			continue ;
		}

		vector2.push_back( kp1 ) ;

		int row1 = (int) ( kp1.irow * 1.0 / kp1.rows * 100 ) ;
		int col1 = (int) ( kp1.icol * 1.0 / kp1.cols * 100 ) ;

		for(int j = i ; j<n1 ; ++ j ){
			wfKeypoint kp2 = vector1[j] ;

			if( vector1[j].flag == -1 ) {
				continue ;
			}

			int row2 = (int) ( kp2.irow * 1.0 / kp2.rows * 100 ) ;
			int col2 = (int) ( kp2.icol * 1.0 / kp2.cols * 100 ) ;
			
			if( row1==row2 && col1 == col2 ){
				vector1[j].flag = -1 ;
			}

		}

	}

	vector1.clear() ;
	while( vector2.size() > 0 ){
		vector1.push_back( vector2.back() ) ;
		vector2.pop_back() ;
	}

}

void burnNwriteKpToImage( const char* path, Mat grayu , 
	std::vector<wfKeypoint>& vec , int rad ){

	Mat grayout = grayu ;
	int rows0 = grayu.rows ;
	int cols0 = grayu.cols ;
	for( int i = 0 ; i<vec.size() ; ++ i ){
		wfKeypoint kp = vec[i] ;
		int row1 = (int) ( kp.irow * 1.0 / kp.rows * rows0 ) ;
		int col1 = (int) ( kp.icol * 1.0 / kp.cols * cols0 ) ;

		for(int i = row1-rad ; i<=row1+rad ; ++ i  ){
			for(int j = col1-rad ; j<=col1+rad ; ++ j ){
				if( i<0 || j<0 || i>rows0-1 || j>cols0-1 ){
					continue ;
				}else{
					grayout.at<uchar>(i,j) = 255 ;
				}
			}
		}
	}

	imwrite( path , grayout ) ;

}



void burnNwriteKpToImageD( const char* path, Mat grayd , 
	std::vector<wfKeypoint>& vec , int rad ){

	Mat grayout( grayd.rows, grayd.cols , CV_8U ) ;
	int rows0 = grayout.rows ;
	int cols0 = grayout.cols ;
	for( int i = 0 ; i<vec.size() ; ++ i ){
		wfKeypoint kp = vec[i] ;
		int row1 = (int) ( kp.irow * 1.0 / kp.rows * rows0 ) ;
		int col1 = (int) ( kp.icol * 1.0 / kp.cols * cols0 ) ;

		for(int i = row1-rad ; i<=row1+rad ; ++ i  ){
			for(int j = col1-rad ; j<=col1+rad ; ++ j ){
				if( i<0 || j<0 || i>rows0-1 || j>cols0-1 ){
					
				}else{
					grayout.at<uchar>(i,j) = 255 ;
				}
			}
		}
	}

	imwrite( path , grayout ) ;

}




void writeKeypointsToFile( const char* path , std::vector<wfKeypoint>& vec ){
	FILE* pf = fopen( path , "w" ) ;
	fprintf(pf,"rows,cols\n") ;
	fprintf(pf, "%d,%d\n", 128 , (int)vec.size() );
	fprintf(pf, "data\n" );
	for(int i = 0 ; i<128 ; ++ i ){
		for(int j = 0 ; j<vec.size() ; ++ j ){
			fprintf(pf,"%8.5f " , vec[j].desc[i] ) ;
		}
		fprintf(pf, "\n");
	}
	fclose(pf) ;
	pf = NULL ;
}


void wfsiftDetector( const char* inputpath , int numkps ){
	Mat img0 = imread( inputpath  , 1) ;
	Mat gray ;
	cvtColor( img0 , gray , CV_BGR2GRAY ) ;
	Mat grayd = convertdouble(gray) ;
	Mat oct0 = grayd ;
	/*
	GaussianBlur( grayd, grayd , Size(0,0) , 0.5 ) ;
	Mat oct0( grayd.rows*2 , grayd.cols*2 , CV_64F ) ;
	pyrUp( grayd , oct0 , Size(grayd.rows*2,grayd.cols*2) ) ;
	GaussianBlur( oct0, oct0 , Size(0,0) , 1.0 ) ;*/

	std::vector<wfKeypoint> wfkpVector ;

	double sig0 = sqrtf(2.0) ;
	double sig1 = pow(2.0,0.5) * sig0 ;
	double sig2 = pow(2.0,0.5) * sig1 ;
	double sig3 = pow(2.0,0.5) * sig2 ;

	double sigarr[] = { sig0 , sig1 , sig2 , sig3 } ;

	Mat oct1 , oct2 , oct3 ;
	pyrDown(oct0 , oct1 ) ;
	pyrDown(oct1 , oct2 ) ;
	pyrDown(oct2 , oct3 ) ;

	wfsiftComputeOctave( oct0  , sigarr , wfkpVector ) ;
	wfsiftComputeOctave( oct1  , sigarr , wfkpVector ) ;
	wfsiftComputeOctave( oct2  , sigarr , wfkpVector ) ;
	wfsiftComputeOctave( oct3  , sigarr , wfkpVector ) ;
	
	wfsiftRemoveRepeatKeyPoints( wfkpVector ) ;
	printf("Remove repeated kp, then good kp num:%d\n" , 
		(int)wfkpVector.size() ) ;



	char burnpath[100] ;
	sprintf( burnpath , "%s.wfsiftimg.png" , inputpath ) ;
	burnNwriteKpToImage( burnpath ,gray , wfkpVector, 2 ) ;

	char descpath[100] ;
	sprintf( descpath , "%s.wfsiftdesc.txt" , inputpath ) ;
	writeKeypointsToFile( descpath , wfkpVector ) ;
}


void test101findkp( Mat dog , std::vector<wfKeypoint>& v ){
	for(int i = 1 ; i<dog.rows-1 ; ++ i ){
		for(int j = 1 ; j<dog.cols-1 ; ++ j ){

			double p0 = (int)dog.at<double>(i-1,j-1) ;
			double p1 = (int)dog.at<double>(i  ,j-1) ;
			double p2 = (int)dog.at<double>(i+1,j-1) ;

			double p3 = (int)dog.at<double>(i-1,j  ) ;
			double p4 = (int)dog.at<double>(i  ,j  ) ;
			double p5 = (int)dog.at<double>(i+1,j  ) ;

			double p6 = (int)dog.at<double>(i-1,j+1) ;
			double p7 = (int)dog.at<double>(i  ,j+1) ;
			double p8 = (int)dog.at<double>(i+1,j+1) ;

			


			if( p4 > p0 && p4 > p1 && p4>p2 && p4 > p3 && 
				p4 > p5 && p4 > p6 && p4>p7 && p4>p8  ){
				double cur = dog.at<double>(i,j) ;
				double dxx = dog.at<double>(i,j-1)
						+ dog.at<double>(i,j+1)
						- 2*cur ;
				double dyy = dog.at<double>(i-1,j)
					+ dog.at<double>(i+1,j)
					- 2*cur ;
				double  dxy = (dog.at<double>(i-1,j-1)
					+ dog.at<double>(i+1,j+1)
					- dog.at<double>(i-1,j+1)
					- dog.at<double>(i+1,j-1))/4.0 ;
				double trH = dxx + dyy;
				double detH = dxx*dyy - dxy*dxy;
				double cim = detH - trH * trH * 0.04 ;
				double curvature_ratio = trH*trH/detH;

				/*
				printf("*** point %d,%d \n" , i,j) ;
				printf("maybe kp! detH %6.2f , cim %6.2f* , curv %6.2f \n" , detH,cim,curvature_ratio ) ;
				printPixelArounded("dogx",dog,i,j,5) ;
				printf("\n") ;
				getchar() ;
				getchar() ;*/
				

				if(detH>0 && cim > CIM_THRESHOLD && cur >8.0 ) {
					wfKeypoint kp ;
					kp.icol = j; 
					kp.irow = i;
					kp.cim = cim ;
					kp.rows = dog.rows ;
					kp.cols = dog.cols ;
					v.push_back(kp) ;
					/*
					printf("kp cur %d,%d : %3.0f \n" , i,j, p4) ;
					printf("%3.0f %3.0f %3.0f %3.0f %3.0f %3.0f %3.0f %3.0f \n" , 
						p0,p1,p2,p3,p5,p6,p7,p8) ;*/
				}



				
			}else if( p4 < p0 && p4 < p1 && p4< p2 && p4 < p3 && 
				p4 < p5 && p4 < p6 && p4 < p7 && p4 < p8 ){
				wfKeypoint kp ;
				kp.icol = j; 
				kp.irow = i;
				kp.rows = dog.rows ;
				kp.cols = dog.cols ;
				//v.push_back(kp) ;
			}

		}
	}
}

void printKeyPoints2File(const char* path , std::vector<wfKeypoint> v , int origsize){
	FILE* pf = fopen(path , "w") ;
	fprintf(pf,"# x y cim \n") ;
	for(int i = 0 ; i<v.size() ; ++ i ){
		fprintf(pf,"%3d %3d %3d %6.1f\n" , i , 
			(int)(v[i].icol*1.0/v[i].cols*origsize) , 
			(int)(v[i].irow*1.0/v[i].rows*origsize) ,
			v[i].cim ) ;
	}
	fclose(pf) ;
	pf = NULL ;
}

void reorderKpsByCim( std::vector<wfKeypoint>& v ){

	for(int i = 0 ; i<v.size() ; ++ i ){
		for(int j = i+1 ; j<v.size() ; ++ j ){
			if( v[i].cim < v[j].cim ) {
				wfKeypoint temp = v[i] ;
				v[i] = v[j] ;
				v[j] = temp ;
			}
		}
	}

}




int main( int argc , char** argv ) {
	/// 

	printf("\n please input mat file path.\n") ;
	char inputpath[100] ;
	scanf("%s" , inputpath ) ;
	int maxNumKps = 0 ;
	printf("\n please input max num keypoint.\n") ;
	scanf("%d" , &maxNumKps ) ;
	if( maxNumKps<=0 ){
		maxNumKps = 10 ;
	}


	Mat mat0 = read2dMatFromFile(inputpath) ;
	//print2dMatFloat2Int(mat0) ;
	Mat blur0 , blur1 , blur2 , blur3 , blur4 ;

	double sig0 = 1.414 ;
	double sig1 = sig0 * 1.6 ;
	double sig2 = sig1 * 1.6 ;
	double sig3 = sig2 * 1.6 ;
	double sig4 = sig3 * 1.6 ;

	
	GaussianBlur(mat0 , blur0 , Size(0,0) , sig0  );

	GaussianBlur(blur0 , blur1 , Size(0,0) , sig1  );
	GaussianBlur(blur0 , blur2 , Size(0,0) , sig2 );
	GaussianBlur(blur0 , blur3 , Size(0,0) , sig3 );
	GaussianBlur(blur0 , blur4 , Size(0,0) , sig4 );

	/*
	print2dMatFloat2Int(blur0) ;
	print2dMatFloat2Int(blur1) ;
	print2dMatFloat2Int(blur2) ;
	print2dMatFloat2Int(blur3) ;
	print2dMatFloat2Int(blur4) ;*/

	Mat dog0 = abs(blur1 - blur0) ;
	Mat dog1 = abs(blur2 - blur1) ;
	Mat dog2 = abs(blur3 - blur2) ;
	Mat dog3 = abs(blur4 - blur3) ;
	/*
	print2dMatFloat2Int(dog0) ;
	print2dMatFloat2Int(dog1) ;
	print2dMatFloat2Int(dog2) ;
	print2dMatFloat2Int(dog3) ;*/


	printf("*** kp ***\n") ;
	std::vector<wfKeypoint> kpvector ;
	test101findkp( dog0 , kpvector ) ;
	test101findkp( dog1 , kpvector ) ;
	test101findkp( dog2 , kpvector ) ;
	test101findkp( dog3 , kpvector ) ;

	printf(" kp num has repeat: %d \n" , (int)kpvector.size() ) ;
	wfsiftRemoveRepeatKeyPoints(kpvector) ;
	//print2dMatFloat2Int(matout) ;
	printf(" kp num remove repeat: %d \n" , (int)kpvector.size() ) ;
	
	reorderKpsByCim( kpvector ) ;
	printKeyPoints2File("keypointvector.txt" , kpvector , 256 ) ;

	std::vector<KeyPoint> v2 ;
	for(int i = 0 ; i< kpvector.size() ; ++ i ){
		if( i >= maxNumKps ) break; 
		KeyPoint kp( kpvector[i].icol*1.0/kpvector[i].cols*256 , 
			kpvector[i].irow*1.0/kpvector[i].rows*256 , 2 ) ;
		v2.push_back(kp) ;
	}
	Mat rgbimage = imread( "t.jpg" , 1 ) ;
	Mat imgout ;
	drawKeypoints( rgbimage , v2 , imgout ) ;
	imwrite("rgbwithkp.png" , imgout) ;

	return 0 ;
}



/*
char filepath1[100] ;
		sprintf( filepath1 , "hi_imgs/img_%d.png" , i ) ;
		FILE* fileok = fopen( filepath1 , "r" ) ;
		if( fileok == NULL ){
			printf( " %d is out " , i ) ;
			break ;
		}else{
			fclose(fileok) ;
		}

		Mat mImage = imread( filepath1  , 1) ;
		cvtColor( mImage , mImage , CV_BGR2GRAY ) ;
		std::vector<KeyPoint> mkps ;
		detector.detect( mImage , mkps ) ;
		Mat mdesc ;    
  		detector.compute( mImage , mkps, mdesc );


//-- Draw only "good" matches
		Mat img_matches;
		drawMatches( inputImage , inputKps, mImage , mkps ,
		       gmatches , img_matches, Scalar::all(-1), Scalar::all(-1),
		       vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		//-- Show detected matches
		char outname[100] ;
		sprintf( outname , "hi_imgs/mat_%d.png" , i ) ;
		imwrite( outname , img_matches ) ;






Mat inputImage = imread( argv[1] , 1) ;
	namedWindow( "input" , WINDOW_AUTOSIZE ) ;
	imshow( "input" , inputImage ) ;

	cvtColor( inputImage , gray , CV_BGR2GRAY) ;
	SiftFeatureDetector detector ;
	std::vector<KeyPoint> keypoints ;
	detector.detect( gray , keypoints ) ;
	Mat siftout ;
	drawKeypoints( inputImage , keypoints , siftout ) ;
	imshow("input" , siftout) ;


src = imread(argv[1],1); 
	namedWindow( "orig" , WINDOW_AUTOSIZE ) ;
	imshow( "orig" , src ) ;

	namedWindow( "out" , WINDOW_NORMAL ) ;

	//strech 0-255
	Mat stretch ;
	normalize( src , stretch , 0 , 255 , CV_MINMAX ) ;
	imshow( "out" , stretch ) ;
	waitKey() ;


	Mat dgray ;
	cvtColor( stretch , dgray, CV_RGB2GRAY ) ;
	imshow( "out" , dgray ) ;
	waitKey() ;
	cv:Scalar tmean = mean(dgray) ;
	double gmean = tmean.val[0] ;


	Mat g01 ;
	g01 = dgray >= gmean ;
	imshow( "out" , g01 ) ;
	waitKey() ;
	

	// 8x8
	resize( dgray , dst , Size(8,8) , 0 , 0 , CV_INTER_LINEAR ) ;
	imshow( "out" , dst ) ;
*/