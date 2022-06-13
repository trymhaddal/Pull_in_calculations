from sections import *

GP_BEND = 500 #Number of gridpoints in a bend
ANNOTATION_OFFSET = 1.5 #Meters offset from section to annotation text in geometry plots
#test
RHO_WET_DEFAULT = 50
RHO_DRY_DEFAULT = 100
MU_WET_PIPE_DEFAULT = 0.25 #[-]
MU_DRY_PIPE_DEFAULT = 0.3
MU_WET_ROLLER_DEFAULT = 0.05
MU_DRY_ROLLER_DEFAULT = 0.1
T_INPUT_DEFAULT = 10000

class GlobalVariables:
    def __init__(self, rho_wet=RHO_WET_DEFAULT, rho_dry=RHO_DRY_DEFAULT, 
                 mu_wet_pipe=MU_WET_PIPE_DEFAULT, mu_dry_pipe=MU_DRY_PIPE_DEFAULT, 
                 mu_wet_roller=MU_WET_ROLLER_DEFAULT, mu_dry_roller=MU_DRY_ROLLER_DEFAULT,
                 T_input=T_INPUT_DEFAULT):
        self.rho_wet = rho_wet
        self.rho_dry = rho_dry
        self.mu_wet_pipe = mu_wet_pipe
        self.mu_dry_pipe = mu_dry_pipe
        self.mu_dry_roller = mu_dry_roller
        self.mu_wet_roller = mu_wet_roller
        self.inputTension = T_input

class Pull_in:
    def __init__(self, sections = [], globalVars=GlobalVariables()):
        #self.inputTension = inputTension
        self.sections = sections
        self.globalVars = globalVars
        self.p0s = self.globalCoordinates()
        self.L,self.T = self.T_L()
        self.endYaw = self.getEndYaw()
        #self.rho_wet = RHO_WET_DEFAULT
        #self.rho_dry = RHO_DRY_DEFAULT
        #self.mu_wet_pipe = MU_WET_PIPE_DEFAULT
        #self.mu_dry_pipe = MU_DRY_PIPE_DEFAULT
        #self.mu_dry_roller = MU_DRY_ROLLER_DEFAULT
        #self.mu_wet_roller = MU_WET_ROLLER_DEFAULT
        #self.inputTension = T_INPUT_DEFAULT


    def addSection(self, s):
        self.sections = np.append(self.sections,s)
        self.p0s = self.globalCoordinates()
        self.L,self.T = self.T_L()
        self.endYaw = self.getEndYaw()

    def updateGlobalVariables(self):
        for sec in self.sections:
            # sec.rho = determineWeight(sec.isWet)
            # sec.mu = determineFrictionFactor(sec.isWet,sec.type)
            sec.rho = determineWeight(sec.isWet,self.globalVars)
            sec.mu = determineFrictionFactor(sec.isWet,sec.surfType,self.globalVars)
        self.__init__(globalVars=self.globalVars,sections=self.sections)
    
    def updateSection(self,sec,index):
        self.sections[index] = sec
        self.__init__(globalVars=self.globalVars,sections=self.sections)

    def deleteSection(self,index):
        newSecs = np.delete(self.sections,index)
        self.__init__(globalVars=self.globalVars,sections=newSecs)

    def getYawDiffs(self):
        if len(self.sections) > 0:
            dPsiVec = np.zeros(len(self.sections)) #
            isHorBend = [(type(self.sections[i]) == HorizontalBend) for i in range(len(self.sections))]
            for i,s in enumerate(self.sections): #Calculating out-in angle in horizontal direction
                if isHorBend[i]: dPsiVec[i] = s.psi_out - s.psi_in
                else: dPsiVec[i] = 0
            return dPsiVec
        else:
            return [0]

    def getOutputYawData(self): #Calculates vector of input yaw for each section by summnig horizontal section contributions
        return np.cumsum(self.getYawDiffs())
    
    def getInputYawData(self):
        return self.getOutputYawData() - self.getYawDiffs() #By definition of deltaPsi

    def getEndYaw(self):
        psiVec = self.getOutputYawData()
        return psiVec[-1]
    
    def updateYawData(self): #Updates input yaw for all sections
        psiInVec = self.getInputYawData()
        psiOutVec = self.getOutputYawData()
        for i,s in enumerate(self.sections):
            s.csYaw = psiInVec[i]
            s.psi_in = psiInVec[i]
            s.psi_out = psiOutVec[i]
            if isinstance(s,HorizontalBend): s.psiVec = np.linspace(s.psi_in,s.psi_out,GP_BEND) #Update all points in psiVec
    
    def updateGeometry(self):
        self.updateYawData()
        [s.updateGeometry() for s in self.sections] #Updates local geometry
        self.p0s = self.globalCoordinates()
        #[s.updateGeometry() for s in self.sections]

    def sectionSeparatorLengths(self):
        individualLenghts = np.array([s.length() for s in self.sections]) #use length() function for every element in sections
        individualLenghts = np.concatenate([[0],individualLenghts])
        return np.cumsum(individualLenghts)
    
    def calculateMinWinchTension(self):
        T_cum = self.inputTension #Cumulative tension required
        for s in self.sections:
            T_cum = s.getMinOutputTension(T_cum)
        return T_cum

    def T_L(self): #Caclulate minimum tension along cable in sequence
        L_vec_global = np.array([0])
        T_vec_global = np.array([self.globalVars.inputTension]) #Cumulative tension required
        self.T_inputs = np.array(np.zeros(len(self.sections))) #Tension at section inputs
        self.T_outputs = np.array(np.zeros(len(self.sections))) #Tension at section outputs
        for i,s in enumerate(self.sections):
            L_vec_global = np.concatenate([L_vec_global,L_vec_global[-1] + s.getLvec()])
            T_input = T_vec_global[-1]
            self.T_inputs[i] = T_input
            T_sec = s.T_vec(T_input)
            self.T_outputs[i] = T_sec[-1]
            T_vec_global = np.concatenate([T_vec_global,T_sec])
        
        return L_vec_global,T_vec_global

    def globalCoordinates(self): #Compute startpoint of all sections
        zeroVec = np.zeros(3)
        p0s = [zeroVec for i in range(len(self.sections))]
        for i in range(1,len(self.sections)):
            p0s[i] = p0s[i-1] + self.sections[i-1].getRotatedEndPoint()
        return p0s

    def plotSideGeometry(self): #From side
        index = 0
        for s in self.sections:
            if (index % 2) == 0: isOdd = False
            else: isOdd = True #Odd number section
            s.plotSideGeometry(self.p0s[index],isOdd) #Call local plotting functon with global coordinates
            #if self.annotateIndex: s.annotateSideIndex(self.p0s[index],index+1)
            index = index + 1
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')

    def annotateSide(self, annotationOffset):
        for index,s in enumerate(self.sections):
            s.annotateSideIndex(self.p0s[index],index+1,annotationOffset=annotationOffset)
    
    def annotateTop(self, annotationOffset):
        for index,s in enumerate(self.sections):
            s.annotateTopIndex(self.p0s[index],index+1,annotationOffset=annotationOffset)
    
    def plotSideCenters(self):
        for index,s in enumerate(self.sections):
            if (type(s)==BottomBend) or (type(s)==TopBend):
                s.plotSideCenter(self.p0s[index])
    
    def plotTopCenters(self):
        for index,s in enumerate(self.sections):
            if type(s)==HorizontalBend:
                s.plotTopCenter(self.p0s[index])
    
    def plotSideArrows(self):
        self.sections[0].plotInputTensionArrowSide(self.p0s[0]) #Input tension vector
        self.sections[-1].plotOutputTensionArrowSide(self.p0s[-1]) #Output tension vector. Inputtension for scale

    def plotTopGeometry(self): #From side
        #plt.figure(3)
        index = 0
        for s in self.sections:
            if (index % 2) == 0: isOdd = False
            else: isOdd = True #Odd number section
            s.plotTopGeometry(self.p0s[index],isOdd) #Call local plotting functon with global coordinates
            index = index + 1
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
    
    def plotTopArrows(self):
        self.sections[0].plotInputTensionArrowTop(self.p0s[0]) #Input tension vector
        self.sections[-1].plotOutputTensionArrowTop(self.p0s[-1]) #Output tension vector. Inputtension for scale
    
    def plot3dGeometry(self):
        plt.figure(5)
        ax = plt.axes(projection='3d')
        index = 0
        for s in self.sections:
            if (index % 2) == 0: col = 'c' #Even number section
            else: col = 'b' #Odd number section
            s.plot3dGeometry(self.p0s[index],ax,col) #Call local plotting functon with global coordinates
            index = index + 1
        #self.sections[0].plotInputTensionArrowTop(self.p0s[0]) #Input tension vector
        #self.sections[-1].plotOutputTensionArrowTop(self.p0s[-1]) #Output tension vector. Inputtension for scale
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

    def tensionHangofPoints(self):
        L_points = np.array([])
        sepLengths = self.sectionSeparatorLengths()
        for i in range(len(self.sections)):
            s = self.sections[i]
            if isinstance(s,BottomBend):
                L_points = np.concatenate([L_points,sepLengths[i] + s.critLenghts()])
        return L_points
    
    def plotMinTensionAlongCable(self):
        #print(len(self.L),len(self.T))
        plt.figure(3)
        plt.plot(self.L,self.T/1000,'k')
        plt.xlabel('Length along cable [m]')
        plt.ylabel('Tension [kN]')
        [plt.axvline(L_ho, color='g',linestyle='-.') for L_ho in self.tensionHangofPoints()] #Plot vertical lines for hangoffs
        [plt.axvline(sepLen, color='k',linestyle='--') for sepLen in self.sectionSeparatorLengths()] #Plot vertical sep lines
        for L_ho in self.tensionHangofPoints():
            text(L_ho, max(self.T)*0.98, r'$L_{ho}$ = %.2f' % L_ho, rotation=90, va='top', ha='right',color='g') #Text for hangoff point vlines
        plt.title('Tension along cable')
