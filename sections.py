#from msilib.schema import Error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
from enum import Enum

g = 9.81 # [m/s]
GP_BEND = 500 #Number of gridpoints in a bend
ANNOTATION_OFFSET = 1.5 #Meters offset from section to annotation text in geometry plots

class SurfaceType(Enum):
    PIPE = 1
    ROLLER = 2

class UnitType(Enum):
    METRIC = 1
    BOTH = 2

class CoordinateSystemType(Enum):
    CARTESIAN = 1
    POLAR = 2

class TableOrientation(Enum):
    HORIZONTAL = 1
    VERTICAL = 2

def determineFrictionFactor(isWet,surfType, globalVars):
    if (isWet):
        if surfType == SurfaceType.PIPE: return globalVars.mu_wet_pipe
        elif surfType == SurfaceType.ROLLER: return globalVars.mu_wet_roller
    else:
        if surfType == SurfaceType.PIPE: return globalVars.mu_dry_pipe
        elif surfType == SurfaceType.ROLLER: return globalVars.mu_dry_roller

def determineWeight(isWet, globalVars):
    if (isWet): return globalVars.rho_wet
    else: return globalVars.rho_dry

def rotate2global(vec,yaw,pitch): #Rotating a vector from local coord system to global coordinate system with rotation around y-axis(pitch) and z-axis(yaw)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0] , [0, 0, 1]]) #3D Rotation matrix
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0] , [-sp, 0, cp]]) #3D Rotation matrix
    return R_yaw @ R_pitch @ vec

def printableRadius(rad):
    if np.isinf(rad): return '\u221e' #Unicode infinity symbol
    else: return rad

def normalizeVec(vec):
    return vec/np.linalg.norm(vec)

def projectVector(vec,dim): #dim: x=0,y=1,z=2
    return np.dot(vec,unitVector(dim))*unitVector(dim)

def unitVector(dim):
    if dim==0:   return np.array([1,0,0])
    elif dim==1: return np.array([0,1,0])
    elif dim==2: return np.array([0,0,1])
 #   else: Error('dim must be 0, 1 or 2')
    
class Section:
    def __init__(self,isWet,surfType,globalVars):
        self.isWet = isWet
        self.surfType = surfType
        self.unrotatedPoints = self.getUnrotatedGeometry() #Local vector of 3D points in reference to unrotated coordinate system
        self.rotatedPoints = self.getRotatedGeometry() #Local vector of 3D points in reference to rotated coordinate system to match world coords
        self.rho = determineWeight(isWet,globalVars=globalVars)
        self.mu = determineFrictionFactor(isWet,surfType,globalVars=globalVars)
        self.inputTension = None #To be determined by Pull-in class

    def length(self):
        pass

    def plotSideGeometry(self,p0,isOddSection): #p0 is global coordinate of local coordinate system
        plt.figure(1)
        globalPoints = p0 + self.rotatedPoints
        xVecGlobal = [p[0] for p in globalPoints]
        zVecGlobal = [p[2] for p in globalPoints]
        plt.plot(xVecGlobal ,zVecGlobal,color=self.plotColor(isOddSection), ls=self.plotStyle()) #From side
        plt.plot(xVecGlobal[0],zVecGlobal[0], marker= '.', color = 'k' )
        plt.plot(xVecGlobal[-1],zVecGlobal[-1], marker= '.', color = 'k')
        plt.title('Geometry side view')
    
    def plotTopGeometry(self,p0,isOddSection):
        plt.figure(2)
        globalPoints = p0 + self.rotatedPoints
        xVecGlobal = [p[0] for p in globalPoints]
        yVecGlobal = [p[1] for p in globalPoints]
        plt.plot(xVecGlobal ,yVecGlobal, color=self.plotColor(isOddSection),ls=self.plotStyle()) #From top
        plt.plot(xVecGlobal[0], yVecGlobal[0], marker= '.', color = 'k' )
        plt.plot(xVecGlobal[-1], yVecGlobal[-1], marker= '.', color = 'k')
        plt.title('Geometry top view')

    def plot3dGeometry(self,p0,ax,col):
        globalPoints = p0 + self.rotatedPoints
        xVecGlobal = [p[0] for p in globalPoints]
        yVecGlobal = [p[1] for p in globalPoints]
        zVecGlobal = [p[2] for p in globalPoints]
        ax.plot3D(xVecGlobal, yVecGlobal, zVecGlobal, color=col) #From top
        #ax.view_init(elev=90, azim=0)

    def getUnrotatedGeometry(): #returns x and y point in local coordinates
        pass
    
    def getRotatedGeometry(self):
        zeroVec = np.zeros(3)
        rotPoints = [zeroVec for i in range(len(self.unrotatedPoints))]
        for i,p in enumerate(self.unrotatedPoints):
            rotPoints[i] = rotate2global(p,self.csYaw,self.csPitch)
        return rotPoints
    
    def getRotatedEndPoint(self):
        return self.rotatedPoints[-1] #Overwritten in TopBend class

    def updateGeometry(self):
        self.unrotatedPoints = self.getUnrotatedGeometry()
        self.rotatedPoints = self.getRotatedGeometry()

    def minOutputTension():
        pass

    def getLvec():
        pass

    def T_vec():
        pass
    
    def getUnitInputVec(self):
        u = self.rotatedPoints[0] - self.rotatedPoints[1]
        return u/np.linalg.norm(u)

    def plotInputTensionArrowSide(self,p0):
        u_norm = (self.rotatedPoints[0] - self.rotatedPoints[1])/(np.linalg.norm(self.rotatedPoints[0] - self.rotatedPoints[1]))
        xlimRange = plt.gca().get_xlim()
        ylimRange = plt.gca().get_ylim()
        scl = 0.1*max(xlimRange[1]-xlimRange[0],ylimRange[1]-ylimRange[0])
        plt.arrow(p0[0],p0[2],scl*u_norm[0],scl*u_norm[2], head_width = scl*0.2, color='k') #Plot in opposite direction of second point
        #plt.text(p0[0] + 1.5*scl*u_norm[0], p0[2] + 1.5*scl*u_norm[2], r'$T_{in}$',va='center',ha='center')
    
    def plotOutputTensionArrowSide(self,p0):
        u_norm = (self.rotatedPoints[-1] - self.rotatedPoints[-2])/(np.linalg.norm(self.rotatedPoints[-1] - self.rotatedPoints[-2]))
        pEnd = p0 + self.rotatedPoints[-1]
        xlimRange = plt.gca().get_xlim()
        ylimRange = plt.gca().get_ylim()
        scl = 0.2*max(xlimRange[1]-xlimRange[0],ylimRange[1]-ylimRange[0])
        plt.arrow(pEnd[0],pEnd[2],scl*u_norm[0],scl*u_norm[2],  head_width = scl*0.1, color='k') #Plot in opposite direction of second point
        #plt.text(pEnd[0] + 1.25*scl*u_norm[0], pEnd[2] + 1.25*scl*u_norm[2], r'$T_{out}$',va='center',ha='center')

    def plotInputTensionArrowTop(self,p0):
        u_norm = (self.rotatedPoints[0] - self.rotatedPoints[1])/(np.linalg.norm(self.rotatedPoints[0] - self.rotatedPoints[1]))
        xlimRange = plt.gca().get_xlim()
        ylimRange = plt.gca().get_ylim()
        scl = 0.1*max(xlimRange[1]-xlimRange[0],ylimRange[1]-ylimRange[0])
        plt.arrow(p0[0],p0[1],scl*u_norm[0],scl*u_norm[1],  head_width = scl*0.2, color='k') #Plot in opposite direction of second point
        #plt.text(p0[0] + scl*u_norm[0],p0[1] + scl*u_norm[1], r'$T_{in}$',va='bottom',ha='right')
    
    def plotOutputTensionArrowTop(self,p0):
        u_norm = (self.rotatedPoints[-1] - self.rotatedPoints[-2])/(np.linalg.norm(self.rotatedPoints[-1] - self.rotatedPoints[-2]))
        pEnd = p0 + self.rotatedPoints[-1]
        xlimRange = plt.gca().get_xlim()
        ylimRange = plt.gca().get_ylim()
        scl = 0.2*max(xlimRange[1]-xlimRange[0],ylimRange[1]-ylimRange[0])
        plt.arrow(pEnd[0],pEnd[1],scl*u_norm[0],scl*u_norm[1],  head_width = scl*0.1, color='k') #Plot in opposite direction of second point
        #plt.text(pEnd[0] + scl*u_norm[0],pEnd[1] + scl*u_norm[1], r'$T_{out}$',va='bottom',ha='right')
    
    def getOutputAngle(self):
        pass
    
    def plotColor(self, isOddSectionNo):
        # if self.isWet:
        #     if isOddSectionNo: return 'blue'
        #     else: return 'royalblue'
        # else: 
        #     if isOddSectionNo: return 'green'
        #     else: return 'limegreen'
        if self.isWet: return 'blue'
        else: return 'deepskyblue'
    
    def plotStyle(self):
        if self.surfType == SurfaceType.PIPE: return '-'
        elif self.surfType == SurfaceType.ROLLER: return ':'
        else: raise NameError('Invalid SectionType')

    def getSectionTypeString(self):
        if isinstance(self,Straight): return 'Straight'
        elif isinstance(self,BottomBend): return 'Bottom bend'
        elif isinstance(self,TopBend): return 'Top bend'
        elif isinstance(self,HorizontalBend): return 'Horizontal bend'

    def annotateSideIndex(self, p0, index):
        pass
    
    def annotateTopIndex(self, p0, index, annotationOffset):
        midpoint = p0 + self.getLocalMidpoint()
        offset = rotate2global(unitVector(dim=1),self.csYaw,self.csPitch) #Rotate y axis
        textpoint = midpoint + annotationOffset*offset
        # projectedMidpoint = projectVector(globalMidpoint,dim=0)
        # normalVec = rotate2global(normalizeVec(projectVector(p0,dim=0)-projectVector(globalPoints[1], dim=0)),-np.pi/2,0)
        # textpoint = projectedMidpoint + annotationOffset*normalVec #Adds a perpendicular vector from midpoint by rotating 90 deg
        #textpoint = self.annotationPoint(p0,annotationOffset)
        plt.annotate(text='{:.0f}'.format(index),xy=(textpoint[0],textpoint[1]),ha='center',va='center')

class Straight(Section):
    def __init__(self,isWet,surfType,horDisp,vertDisp,yaw_in_deg, globalVars):
        self.theta = np.arctan2(vertDisp,horDisp) #Constant pitch. Negative theta corresponds to positive pitch
        self.csPitch = -self.theta #Pitch of local coordinate system
        self.csYaw = np.deg2rad(yaw_in_deg)
        self.horDisp = horDisp
        self.vertDisp = vertDisp
        self.theta_in = self.theta
        self.theta_out = self.theta
        self.psi_in = self.csYaw
        self.psi_out = self.csYaw
        self.R = np.Infinity
        #self.R = r'$\infty$'
        super().__init__(isWet,surfType,globalVars)

    def getUnrotatedGeometry(self):
        p1 = np.array([0,0,0])
        p2 = np.array([self.length(),0,0])
        return [p1,p2] #Return list of 3D point vectors in relation to local coordinate system

    def getLvec(self):
        return 0,self.length()
    
    def length(self):
        return np.sqrt(self.horDisp**2 + self.vertDisp**2)
    
    def minOutputTension(self, inputTension):
        return inputTension + self.rho*self.length()*g*(self.mu*np.cos(self.theta) + np.sin(self.theta))
    
    def T_vec(self,inputTension):
        return inputTension,self.minOutputTension(inputTension)

    def getLocalMidpoint(self):
        return  0.5*(self.rotatedPoints[0]+self.rotatedPoints[1])

    def annotateSideIndex(self, p0, index, annotationOffset):
        midpoint = p0 + self.getLocalMidpoint()
        # normalVec = rotate2global(normalizeVec(self.rotatedPoints[1]-self.rotatedPoints[0]),0,-np.pi/2)
        # textpoint = midpoint + annotationOffset*normalVec #Adds a perpendicular vector from midpoint by rotating 90 deg
        offset = rotate2global(unitVector(dim=2),self.csYaw,self.csPitch) #Rotate z axis
        textpoint = midpoint + annotationOffset*offset
        #textpoint = self.annotationPoint(p0,annotationOffset)
        plt.annotate(text='{:.0f}'.format(index),xy=(textpoint[0],textpoint[2]),ha='center',va='center')

class VerticalBend(Section):
    def __init__(self,isWet,surfType,R,theta_in_deg,theta_out_deg,yaw_in_deg, globalVars):
        self.csYaw = np.deg2rad(yaw_in_deg)
        self.R = R
        self.theta_in = np.deg2rad(theta_in_deg)
        self.theta_out = np.deg2rad(theta_out_deg)
        self.thetaVec = np.linspace(self.theta_in,self.theta_out,GP_BEND)
        super().__init__(isWet,surfType,globalVars) #Calls Section constructor
        self.L_vec = self.getLvec()
        self.psi_in = self.csYaw
        self.psi_out = self.csYaw

    def getUnrotatedGeometry(self):
        pass

    def getLvec(self): #Length along arc as a function of thetaVec
        return ((self.thetaVec-self.theta_in)*self.R)

    def length(self):
        return (self.getLvec()[-1])

    def minOutputTension(self, inputTension):
        T_theta = self.T_vec(inputTension)
        return T_theta[-1]
    
    def T_vec():
        pass
    
    def getRotatedGeometry(self):
        return super().getRotatedGeometry()

    def annotateSideIndex(self, p0, index):
        pass
    
    def getLocalCenter(self):
        pass

    def getLocalMidpoint(self):
        return self.rotatedPoints[int(len(self.rotatedPoints)/2)] #Round up from half

    # def annotateTopIndex(self, p0, index, annotationOffset):
    #     midpoint = p0 + self.getLocalMidpoint()
    #     projectedMidpoint = projectVector(midpoint, dim=0)
    #     projected_p0 = projectVector(p0, dim=0)
    #     normalVec = rotate2global(normalizeVec(projectedMidpoint - projected_p0),np.pi/2,0)
    #     textpoint = projectedMidpoint + annotationOffset*normalVec
    #     plt.annotate(text='{:.0f}'.format(index),xy=(textpoint[0],textpoint[1]),ha='center',va='center')

    def annotationPoint(self, p0, annotationOffset):
        midpoint = p0 + self.getLocalMidpoint()

class BottomBend(VerticalBend):
    def __init__(self,isWet,surfType,R,theta_in_deg,theta_out_deg,psi_in_deg, globalVars):
        self.csPitch = np.deg2rad(-theta_in_deg) #Pitch of local coordinate system¨
        super().__init__(isWet,surfType,R,theta_in_deg,theta_out_deg,psi_in_deg,globalVars=globalVars)
        self.theta_crits = None #To be determined with inputTension from Pull-in class
    
    def getUnrotatedGeometry(self):
        deltaThetaVec = self.thetaVec - self.theta_in
        zeroVec = np.zeros(3)
        unrotPoints = [zeroVec for i in range(len(deltaThetaVec))]
        for i,dTheta in enumerate(deltaThetaVec):
            unrotPoints[i] = np.array([0,0,self.R]) + rotate2global(np.array([0,0,-self.R]),0,-dTheta) #Pitch = -theta
        return unrotPoints

    def getRotatedCenterPoint(self):
        return rotate2global(np.array([0,0,self.R]),self.csYaw,self.csPitch)

    def critLenghts(self):
        valid_theta_crits = self.theta_crits[ (self.theta_crits > self.theta_in) & (self.theta_crits < self.theta_out) ]
        L_crits = ((valid_theta_crits-self.theta_in)*self.R)
        return L_crits[L_crits > 0]

    def T_bottom(self,T_in,theta_in,theta):
        A = self.rho*self.R*g/(self.mu**2+1)
        B = self.mu**2 - 1
        C = self.mu*2
        
        exp_term = np.exp(self.mu*(theta - theta_in))

        T_bottom = T_in*(1/exp_term) + \
                 + A*( B*(np.cos(theta) - np.cos(theta_in)*(1/exp_term)) + \
                       C*(np.sin(theta) - np.sin(theta_in)*(1/exp_term)) )
        return T_bottom
    
    def T_top(self,T_in,theta_in,theta):
        A = self.rho*self.R*g/(self.mu**2+1)
        B = self.mu**2 - 1
        C = self.mu*2
        
        exp_term = np.exp(self.mu*(theta - theta_in))

        T_top = T_in*exp_term + \
                 + A*( B*(np.cos(theta) - np.cos(theta_in)*exp_term ) + \
                       C*(-np.sin(theta) + np.sin(theta_in)*exp_term) )

        return T_top

    def thetaHangoff(self,T_in):
        #N_bottom = lambda theta: self.rho*self.R*g*np.cos(theta) - self.T_bottom(T_in,self.theta_in,theta)
        #roots = fsolve(N_bottom,np.array([-np.pi/8,np.pi/8])) #Find roots from both sides
        #plt.plot(self.L_vec,N_bottom(self.thetaVec))
        thetaRange = np.linspace(-np.pi/2,np.pi/2,GP_BEND)
        N_bottom = self.rho*self.R*g*np.cos(thetaRange) - self.T_bottom(T_in,self.theta_in,thetaRange)
        roots = np.array([])
        
        for i in range(1,len(N_bottom)):
            if (N_bottom[i]*N_bottom[i-1] < 0): 
                roots = np.append(roots,thetaRange[i])
        #roots = fsolve(N_bottom,np.array([-np.pi/8,np.pi/8])) #Find roots from both sides

        return roots
    
    def T_vec(self,inputTension):
        self.theta_crits = self.thetaHangoff(inputTension) #gives two roots which may be outside theta_range
        if len(self.theta_crits) == 0:
            return self.T_top(inputTension,self.theta_in,self.thetaVec)
        elif len(self.theta_crits) == 2:
            isNotResting1 = (self.thetaVec <= self.theta_crits[0])
            isResting = (self.thetaVec > self.theta_crits[0]) & (self.thetaVec < self.theta_crits[1]) #Boolean values for resting theta
            isNotResting2 = (self.thetaVec >= self.theta_crits[1])

            theta_topside1 = self.thetaVec[isNotResting1]
            theta_resting = self.thetaVec[isResting]
            theta_topside2 = self.thetaVec[isNotResting2]

            # Avoid indexing empty array
            if len(theta_topside1) == 0: theta_in_topside1 = []
            else: theta_in_topside1 = theta_topside1[0]

            if len(theta_resting) == 0: theta_in_resting = []
            else: theta_in_resting = theta_resting[0]

            if len(theta_topside2) == 0: theta_in_topside2 = []
            else: theta_in_topside2 = theta_topside2[0]

            T_topside1 = self.T_top(inputTension,theta_in_topside1,theta_topside1)

            if len(T_topside1) == 0: restingInput = inputTension
            else: restingInput = T_topside1[-1]
            T_resting = self.T_bottom(restingInput,theta_in_resting,theta_resting)
            
            if len(T_resting) == 0: topside2Input = inputTension
            else: topside2Input = T_resting[-1]
            T_topside2 = self.T_top(topside2Input,theta_in_topside2,theta_topside2)

            return np.hstack(( T_topside1, T_resting, T_topside2 )).ravel() #Combining vectors into one
        else:
            raise NameError('An unexpected number of roots found for hang-off value')

    def plotSideGeometry(self,p0,col): #Overwrite method in Section class
        super().plotSideGeometry(p0,col)
        valid_theta_crits = self.theta_crits[ (self.theta_crits > self.theta_in) & (self.theta_crits < self.theta_out) ]#Check if theta_crits are in range
        zeroVec = np.zeros(3)
        hangoffPoints = [zeroVec for i in range(len(valid_theta_crits))]
        globalCenter = rotate2global(np.array([0,0,self.R]),self.csYaw,self.csPitch)
        for i,theta_c in enumerate(valid_theta_crits):
            hangoffPoints[i] = p0 + globalCenter + rotate2global(np.array([0,0,-self.R]),self.csYaw,-theta_c)
        hangoffPoints_x = [hp[0] for hp in hangoffPoints]
        hangoffPoints_z = [hp[2] for hp in hangoffPoints]
        plt.plot(hangoffPoints_x ,hangoffPoints_z,'kx') #Hangoff points

    def plotTopGeometry(self,p0,col): #Overwrite method in Section class
        super().plotTopGeometry(p0,col)
        valid_theta_crits = self.theta_crits[ (self.theta_crits > self.theta_in) & (self.theta_crits < self.theta_out) ]#Check if theta_crits are in range
        zeroVec = np.zeros(3)
        hangoffPoints = [zeroVec for i in range(len(valid_theta_crits))]
        globalCenter = rotate2global(np.array([0,0,self.R]),self.csYaw,self.csPitch)

        for i,theta_c in enumerate(valid_theta_crits):
            hangoffPoints[i] = p0 + globalCenter + rotate2global(np.array([0,0,-self.R]),self.csYaw,-theta_c)
        hangoffPoints_x = [hp[0] for hp in hangoffPoints]
        hangoffPoints_y = [hp[1] for hp in hangoffPoints]
        plt.plot(hangoffPoints_x ,hangoffPoints_y,'kx') #Hangoff points

    def getRotatedEndPoint(self):
        return super().getRotatedEndPoint()

    def getLocalCenter(self):
        return rotate2global(np.array([0,0,self.R]),self.csYaw,self.csPitch)

    def annotateSideIndex(self, p0, index, annotationOffset):
        midpoint = p0 + self.getLocalMidpoint()
        globalCenter = p0 + self.getLocalCenter()
        textpoint = midpoint - annotationOffset*normalizeVec(midpoint - globalCenter)
        plt.annotate(text='{:.0f}'.format(index), xy=(textpoint[0],textpoint[2]), ha='center', va='center')

    def plotSideCenter(self, p0):
        globalCenter = p0 + self.getLocalCenter()
        plt.plot(globalCenter[0], globalCenter[2], 'k+')

class TopBend(VerticalBend):
    def __init__(self,isWet,surfType,R,theta_in_deg,theta_out_deg,psi_in_deg, globalVars):
        self.csPitch = np.deg2rad(-theta_in_deg) #Pitch of local coordinate system¨
        super().__init__(isWet,surfType,R,theta_in_deg,theta_out_deg,psi_in_deg,globalVars=globalVars)
    
    def getUnrotatedGeometry(self):
        deltaThetaVec = (self.thetaVec - self.theta_in)
        zeroVec = np.zeros(3)
        unrotPoints = [zeroVec for i in range(len(deltaThetaVec))]
        for i,dTheta in enumerate(deltaThetaVec):
            unrotPoints[i] = np.array([0,0,-self.R]) + rotate2global(np.array([0,0,self.R]),0,-dTheta) #Pitch = -theta
        return unrotPoints
    
    def getRotatedGeometry(self):
        zeroVec = np.zeros(3)
        rotPoints = [zeroVec for i in range(len(self.unrotatedPoints))]
        for i,p in enumerate(self.unrotatedPoints):
            rotPoint = rotate2global(p,self.csYaw,self.csPitch)
            #rotPoints[i] = np.array([ - rotPoint[0], rotPoint[1], rotPoint[2]]) #Mirroring about y axis
            rotPoints[i] = rotate2global(rotPoint,np.pi,0) #Mirror by half turn around z-axis
        return rotPoints
    
    def T_vec(self,inputTension):
        A = self.rho*self.R*g/(self.mu**2+1)
        B = self.mu**2 - 1
        C = self.mu*2
        exp_term = np.exp(self.mu*(self.thetaVec - self.theta_in))

        T = inputTension*(exp_term) + \
                 + A*( B*( -np.cos(self.thetaVec) + np.cos(self.theta_in)*exp_term) + \
                       C*(np.sin(self.thetaVec) - np.sin(self.theta_in)*exp_term) )
        return T

    def getLocalCenter(self):
        return rotate2global(rotate2global(np.array([0,0,-self.R]),self.csYaw,self.csPitch),np.pi,0) #Include mirror flip by rotating pi

    def annotateSideIndex(self, p0, index, annotationOffset):
        midpoint = p0 + self.getLocalMidpoint()
        globalCenter = p0 + self.getLocalCenter()
        textpoint = midpoint + annotationOffset*normalizeVec(midpoint - globalCenter)
        plt.annotate(text='{:.0f}'.format(index), xy=(textpoint[0],textpoint[2]), ha='center',va='center')

    def plotSideCenter(self, p0):
        globalCenter = p0 + self.getLocalCenter()
        plt.plot(globalCenter[0], globalCenter[2], 'k+')

class HorizontalBend(Section):
    def __init__(self,isWet,surfType,R,yaw_in_deg,yaw_out_deg, globalVars):
        self.csPitch = 0
        self.csYaw = np.deg2rad(yaw_in_deg)
        self.psi_in = np.deg2rad(yaw_in_deg)
        self.psi_out = np.deg2rad(yaw_out_deg)
        self.psiVec = np.linspace(self.psi_in,self.psi_out,GP_BEND)
        self.R = R
        self.theta_in = 0
        self.theta_out = 0
        super().__init__(isWet,surfType,globalVars=globalVars)
    
    def getUnrotatedGeometry(self):
        deltaPsiVec = (self.psiVec - self.psi_in)
        zeroVec = np.zeros(3)
        unrotPoints = [zeroVec for i in range(len(deltaPsiVec))]
        if (self.psi_out - self.psi_in > 0): #CounterClockwise turn
            unrotatedCenterPoint = np.array([0,self.R,0]) #In local x'-y' coordinate system
            unrotatedPoint = np.array([0,-self.R,0]) #In local x~ - y~ coordinate system
        else: 
            unrotatedCenterPoint = np.array([0,-self.R,0])
            unrotatedPoint = np.array([0,self.R,0])

        for i,dPsi in enumerate(deltaPsiVec):
            unrotPoints[i] = unrotatedCenterPoint + rotate2global(unrotatedPoint,dPsi,self.csPitch) #Pitch = -theta
        return unrotPoints

    def T_vec(self, inputTension):
        return inputTension*np.exp(self.mu*np.abs(self.psiVec - self.psi_in))

    def getLvec(self): #Length along arc as a function of thetaVec
        return np.abs(((self.psiVec-self.psi_in)*self.R))
    
    def length(self):
        return (self.getLvec()[-1])

    def getLocalMidpoint(self):
        return self.rotatedPoints[int(len(self.rotatedPoints)/2)]

    def getLocalCenter(self):
        return rotate2global(np.array([0,self.R,0]),self.csYaw,self.csPitch)

    def annotateSideIndex(self, p0, index, annotationOffset):
        globalPoints = p0 + self.rotatedPoints
        midpoint = 0.5*(globalPoints[0]+globalPoints[-1])
        #textpoint = midpoint + 0.1*rotate2global(globalPoints[1]-globalPoints[0],0,-np.pi/2) #Adds a perpendicular vector from midpoint by rotating 90 deg
        textpoint = midpoint + annotationOffset*rotate2global(normalizeVec(globalPoints[-1]-globalPoints[0]),0,-np.pi/2) #Adds a perpendicular vector from midpoint by rotating 90 deg
        plt.annotate(text='{:.0f}'.format(index),xy=(textpoint[0],textpoint[2]),ha='center',va='center')
    
    def annotateTopIndex(self, p0, index, annotationOffset):
        midpoint = p0 + self.getLocalMidpoint()
        globalCenter = p0 + self.getLocalCenter()
        textpoint = midpoint - annotationOffset*normalizeVec(midpoint - globalCenter)
        plt.annotate(text='{:.0f}'.format(index),xy=(textpoint[0],textpoint[1]),ha='center',va='center')
    
    def plotTopCenter(self,p0):
        globalCenter = p0 + self.getLocalCenter()
        plt.plot(globalCenter[0],globalCenter[1],'k+')