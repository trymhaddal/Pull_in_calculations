from pull_in import *
from ipywidgets import RadioButtons,BoundedFloatText,IntRangeSlider,IntSlider,Button,BoundedIntText,ToggleButtons\
                        ,FileUpload,VBox,HBox,Layout,Image,Output,Checkbox,Tab,FloatSlider,Text, Accordion
from matplotlib.backends.backend_pdf import PdfPages #For printing to PDF
import functools #In order to have arguments in callback functions
import pandas as pd
from IPython.display import display, clear_output, HTML
from base64 import b64encode
#import fpdf
#import pdfkit

#from ipyupload import FileUpload

wStyle = {'description_width': 'initial'}
box_layout = Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    justify_content = 'space-between',
                    width='100%')

# Output widgets
#region
outText = Output(layout=Layout(flex='1 1 auto'))
outGeoSide = Output(layout = Layout(width = '50%'))
outGeoTop = Output(layout = Layout(width = '50%'))
outTension = Output(layout = Layout(width = '50%'))
outPDF = Output(layout = Layout(width = '20%'))
outDebug = Output(layout = Layout(width = '100%'))
#endregion

# Global parameter widgets
#region
rho_w_BFT = BoundedFloatText(value=RHO_WET_DEFAULT, min=0, max=200, description = 'Wet weight [kg/m]', style = wStyle)
rho_d_BFT = BoundedFloatText(value=RHO_DRY_DEFAULT, min=0, max=200, description = 'Dry weight [kg/m]', style = wStyle)
mu_wp_BFT = BoundedFloatText(value=MU_WET_PIPE_DEFAULT, min=0, max=1, step=0.01, description = 'Wet pipe friction factor [-]', style = wStyle)
mu_dp_BFT = BoundedFloatText(value=MU_DRY_PIPE_DEFAULT, min=0, max=1, step=0.01, description = 'Dry pipe friction factor [-]', style = wStyle)
mu_wr_BFT = BoundedFloatText(value=MU_WET_ROLLER_DEFAULT, min=0, max=1, step=0.01, description = 'Wet roller friction factor [-]', style = wStyle)
mu_dr_BFT = BoundedFloatText(value=MU_DRY_ROLLER_DEFAULT, min=0, max=1, step=0.01, description = 'Dry roller friction factor [-]', style = wStyle)
T_in_BIT = BoundedIntText(value=T_INPUT_DEFAULT, min=0, max=1e6, step=100, description = 'Input tension [N]', style = wStyle)
saveBtn = Button(description='Save parameters', disabled=True, icon='save',button_style='primary')
editGlbBtn = Button(description='Edit parameters', disabled=False, icon='edit',button_style='primary')

weightBox = VBox(children= [rho_w_BFT,rho_d_BFT])
frictionBox = VBox(children = [mu_wp_BFT, mu_dp_BFT, mu_wr_BFT, mu_dr_BFT])
buttonGlbBox = VBox(children = [editGlbBtn,saveBtn])
middleGlobalParamBox = VBox(children = [weightBox, T_in_BIT] )
globalParamTab = HBox(children = [frictionBox, middleGlobalParamBox, buttonGlbBox], layout=box_layout)

globalParamList = [mu_wp_BFT, mu_dp_BFT, mu_wr_BFT, mu_dr_BFT, rho_w_BFT, rho_d_BFT, T_in_BIT] #For easy enabling/disabling
for w in globalParamList: w.disabled = True #Start with all parameters locked into default

for w in globalParamTab.children: w.layout = Layout(width = 'auto', flex = '1 1 auto')
middleGlobalParamBox.layout.flex = '3 1 auto'
middleGlobalParamBox.layout.justify_content = 'space-between' #Must be stated after adjusting flex
#endregion

# Geometry & options widgets
#region
isWet_RB = RadioButtons(options = [('Wet',True),('Dry',False)])
Material_RB = RadioButtons(options = [('Pipe',SurfaceType.PIPE),('Roller',SurfaceType.ROLLER)])
geometry_RB = RadioButtons(options = [('Straight',Straight),('Bottom bend',BottomBend),('Top bend',TopBend),('Horizontal bend',HorizontalBend)])
horDisp_BFT = BoundedFloatText(value=5, min=0, max=2000, step = 1, description = 'Horizontal displacement [m]',style = wStyle)
vertDisp_BFT = BoundedFloatText(value=5, min=-100 , max=100, description = 'Vertical displacement [m]',style = wStyle)
angle_BFT = BoundedFloatText(value=45, min=-90, max=90, step = 1, description = 'Angle [deg]',style = wStyle)
length_BFT = BoundedFloatText(value=5, min=0 , max=100, description = 'Length [m]',style = wStyle)
rad_BFT = BoundedFloatText(value=5, min=0 ,max=2000, description = 'Bend radius [m]',style = wStyle)
thetaRange_FRS = IntRangeSlider(value=[-45, 45], min=-90, max=90, step=1, description = 'Pitch range [deg]',style = wStyle)
psi_IS = IntSlider(value = 0, min=-90, max=90, step=5, description = 'Delta Yaw [deg]', style = wStyle)
addSecBtn = Button(description='Add section', disabled=False, icon='check', button_style='info')
editSecBtn = Button(description='Edit section', disabled=True, icon='edit', button_style='primary')
deleteScnBtn = Button(description='Delete', disabled= True, icon='close', button_style='danger')
updateScnBtn = Button(description='Update', disabled= True, icon='refresh', button_style='primary')
editSecNum_BIT = BoundedIntText(description= 'Section #', disabled = True, value=1, min=1 , max = 1)
#importGeometryBtn = FileUpload(description='Upload geometry', disabled=False, icon='upload',button_style='primary')
geometryUploader = FileUpload(accept='', multiple=False)
inputCoordsTypeTB = ToggleButtons(options = [('Polar',CoordinateSystemType.POLAR), ('Cartesian',CoordinateSystemType.CARTESIAN)])

polarBox = VBox(children= [angle_BFT, length_BFT])
cartesianBox = VBox(children= [horDisp_BFT, vertDisp_BFT])
editButtonBox = VBox(children = [editSecNum_BIT, HBox(children = [updateScnBtn, deleteScnBtn])])
#buttonSecVBox = VBox(children =[addSecBtn,editSecBtn])
editBox = HBox(children = [addSecBtn, editSecBtn, editButtonBox], layout= Layout(border='solid'))
#buttonSecBox = HBox(children = [VBox(children =[buttonSecHBox,geometryUploader ]),editButtonBox])
#buttonSecBox = HBox(children = [buttonSecVBox,editButtonBox])
#buttonSecBox = HBox(children = [addSecBtn, editBox])

editList = [editSecNum_BIT,deleteScnBtn,updateScnBtn]

secVarOptions = VBox(children= [VBox(children= [angle_BFT, length_BFT]), inputCoordsTypeTB]) #Options for adding Straight section
sectionTab = HBox(children = [isWet_RB, Material_RB, geometry_RB, secVarOptions, editBox],
                  style = wStyle, layout=box_layout)

for w in sectionTab.children: w.layout = Layout(width = 'auto', flex = '1 1 auto')
#editBox.layout = Layout(width = '20%', flex = '1 1 auto')
sectionHBox = HBox(children = [isWet_RB, Material_RB, geometry_RB, secVarOptions]) #For use in changeGeometrySpecificationUI

#creatingSecBox = GridspecLayout()
# sectionBox = GridspecLayout(1,9)
# sectionBox[:,:6] = secVarOptions
# sectionBox[:,6:] = editBox

#endregion

# Options widgets
#region
axisEqCB = Checkbox(description='Axes equal', disabled=False, value=True, indent=False)
arrowCB = Checkbox(description='Show arrows', disabled=False, value=True, indent=False)
annotateCB = Checkbox(description='Index Labels', disabled=False, value=True, indent=False)
#saveCsvBtn = Button(description='Save csv', disabled=False, icon='file-text-o', button_style='success')
annotationOffsetFS = FloatSlider(value = 1.0, min=0, max=5, step=0.5, description = 'Text offset', style = wStyle)
units_RB = RadioButtons(options = [('Metric',UnitType.METRIC),('Imperial (metric)',UnitType.BOTH)])
annotationOffsetFS = FloatSlider(description = 'Text offset [m]', value = 1.0, min=-5, max=5, step=0.5, style = wStyle)
tableOrientationRB = RadioButtons(options = [('Horizontal',TableOrientation.HORIZONTAL),('Vertical',TableOrientation.VERTICAL)])
plotCenterCB = Checkbox(description='Plot bend centers', disabled=False, value=False, indent=False)

exportSidePlotCB = Checkbox(description='Side geometry', disabled=False, value=True, indent=False)
exportTopPlotCB = Checkbox(description='Top geometry', disabled=False, value=True, indent=False)
exportTensionPlotCB = Checkbox(description='Tension plot', disabled=False, value=True, indent=False)
exportTableCB = Checkbox(description='Table', disabled=False, value=True, indent=False)

#exportPdfBtn = Button(description='Generate PDF', disabled=False, icon='file-pdf-o', button_style='success')
fileNameTxt = Text(description='Filename', disabled=False, value='pull_in_calc', placeholder='pull_in_calc')
exportFilesBtn = Button(description='Generate files', disabled=False, icon='file-archive-o', button_style='success')

exportCheckBoxes = VBox(children = [exportSidePlotCB, exportTopPlotCB, exportTensionPlotCB, exportTableCB])
plotCheckBoxes = VBox(children = [axisEqCB, arrowCB, plotCenterCB, HBox(children=[annotateCB, annotationOffsetFS])])
#exportBox = VBox(children = [exportBtn,outPDF])
#fileBox = VBox(children=[exportCheckBoxes,exportPdfBtn, exportFilesBtn, outPDF])
fileBox = VBox(children=[fileNameTxt, exportFilesBtn, outPDF])

optionsTab = HBox(children = [tableOrientationRB, units_RB, plotCheckBoxes],
                         layout=Layout(display='flex', flex_flow='row', justify_content='flex-start', align_items='center', flex='0 1 auto'))
plotCheckBoxes.layout = Layout(display = 'flex', justify_content = 'space-between')
for w in plotCheckBoxes.children: w.layout = Layout(width = 'auto', flex = '1 1 auto')
for w in fileBox.children: w.layout = Layout(width = 'auto', flex = '1 1 auto')

exportTab =HBox(children=[exportCheckBoxes, fileBox])
#endregion

# Collect widgets
#region
tabTitles = ['Geometry','Global Parameters','Options','Export']
tab = Tab(children = [sectionTab, globalParamTab, optionsTab, exportTab])
tab.layout.width = '100%'
[tab.set_title(i, title) for i, title in enumerate(tabTitles)]

topBox = HBox(children = [tab])
#middleBox = HBox(children = [outText,resultHandlingBox], layout=box_layout)
bottomBox = HBox(children = [outGeoSide,outGeoTop,outTension])
ui = VBox(children = [topBox, outText, bottomBox])
#ui = VBox(children = [outDebug, topBox, outText, bottomBox])
#display(outDebug)
imgFile = open('nexans_logo.png', 'rb')
imageWidget = Image(value=imgFile.read(), format='png', width=300, height=300)

#endregion

# Functions

def compileSectionFromGUI(pi,index=-1): #Compiles a Section object with widget values
    psiOutVec = pi.getOutputYawData() #In radians
    psiInVec = pi.getInputYawData()
    if index == -1: yaw_in = np.rad2deg(psiOutVec[-1]) #Append to end
    else: yaw_in = np.rad2deg(psiInVec[index])
    if geometry_RB.value == Straight:
        if inputCoordsTypeTB.value == CoordinateSystemType.CARTESIAN:
            sec = Straight(isWet_RB.value, Material_RB.value, horDisp_BFT.value, vertDisp_BFT.value, yaw_in,globalVars=pi.globalVars)
        else:
            l = length_BFT.value
            theta = np.deg2rad(angle_BFT.value)
            sec = Straight(isWet_RB.value, Material_RB.value, l*np.cos(theta), l*np.sin(theta), yaw_in, globalVars=pi.globalVars)
    elif geometry_RB.value == BottomBend: 
        sec = BottomBend(isWet_RB.value,Material_RB.value,rad_BFT.value,thetaRange_FRS.value[0],thetaRange_FRS.value[1], yaw_in, globalVars=pi.globalVars)
    elif geometry_RB.value == TopBend: 
        sec = TopBend(isWet_RB.value,Material_RB.value,rad_BFT.value,thetaRange_FRS.value[0],thetaRange_FRS.value[1], yaw_in, globalVars=pi.globalVars)
    elif geometry_RB.value == HorizontalBend: 
        sec = HorizontalBend(isWet_RB.value, Material_RB.value, rad_BFT.value, yaw_in, yaw_in + psi_IS.value, globalVars=pi.globalVars)
    return sec

def updateSectionOutput(pi):
    df = generateDataframe(pi) if (units_RB.value==UnitType.METRIC) else generateDataframeWithImperial(pi)
    if tableOrientationRB.value==TableOrientation.VERTICAL: df = df.transpose()
    with outText: clear_output()
    with outText: display(df)

def generateDataframe(pi): #Pandas dataframe

    dataDic = {
        'Type'       : [sec.getSectionTypeString() for sec in pi.sections],
        'W/D'        : ['Wet' if sec.isWet else 'Dry' for sec in pi.sections],
        'P/R'        : ['Pipe' if sec.surfType==SurfaceType.PIPE else 'Roller' for sec in pi.sections],
        'R [m]'      : ['\u221e' if np.isinf(sec.R) else '{:.1f}'.format(sec.R) for sec in pi.sections],
        'θ_in [deg]' : ['{:.1f}'.format(np.rad2deg(sec.theta_in)) for sec in pi.sections],
        'θ_out [deg]': ['{:.1f}'.format(np.rad2deg(sec.theta_out)) for sec in pi.sections],
        'Ψ_in [deg]' : ['{:.1f}'.format(np.rad2deg(sec.psi_in)) for sec in pi.sections],
        'Ψ_out [deg]': ['{:.1f}'.format(np.rad2deg(sec.psi_out)) for sec in pi.sections],
        'μ_s [-]'    : ['{:.2f}'.format(sec.mu) for sec in pi.sections],
        'w [kg/m]'   : ['{:.1f}'.format(sec.rho) for sec in pi.sections],
        'L [m]'      : ['{:.1f}'.format(sec.length()) for sec in pi.sections],
        'T_in [kN]'  : ['{:.1f}'.format(T_in/1000) for T_in in pi.T_inputs],
        'T_sec [kN]' : ['{:.1f}'.format((T_out-T_in)/1000) for T_in,T_out in zip(pi.T_inputs,pi.T_outputs)],
        'T_out [kN]' : ['{:.1f}'.format(T_out/1000) for T_out in pi.T_outputs]
    }

    indices = [i for i in range(1,len(pi.sections)+1)]
    #df = pd.DataFrame(data=dataDic, index=indices)
    return pd.DataFrame(data=dataDic, index=indices)

def generateDataframeWithImperial(pi): #Pandas dataframe
    kg2lb = 2.2046 #1 kg to lb
    m2ft = 3.281 #1 m to ft
    N2lbf = 0.2248 #1 N to lbf
    dataDic = {
        'Type'             : [sec.getSectionTypeString() for sec in pi.sections],
        'W/D'              : ['Wet' if sec.isWet else 'Dry' for sec in pi.sections],
        'P/R'              : ['Pipe' if sec.surfType==SurfaceType.PIPE else 'Roller' for sec in pi.sections],
        'R [ft (m)]'       : ['\u221e' if np.isinf(sec.R) else '{:.1f} ({:.1f})'.format(sec.R*m2ft,sec.R) for sec in pi.sections],
        'θ_in [deg]'       : ['{:.1f}'.format(np.rad2deg(sec.theta_in)) for sec in pi.sections],
        'θ_out [deg]'      : ['{:.1f}'.format(np.rad2deg(sec.theta_out)) for sec in pi.sections],
        'Ψ_in [deg]'       : ['{:.1f}'.format(np.rad2deg(sec.psi_in)) for sec in pi.sections],
        'Ψ_out [deg]'      : ['{:.1f}'.format(np.rad2deg(sec.psi_out)) for sec in pi.sections],
        'μ_s [-]'          : ['{:.2f}'.format(sec.mu) for sec in pi.sections],
        'w [lbf/ft (kg/m)]': ['{:.1f} ({:.1f})'.format(sec.rho*kg2lb/m2ft, sec.rho) for sec in pi.sections],
        'L [ft (m)]'       : ['{:.1f} ({:.1f})'.format(sec.length()*m2ft,sec.length()) for sec in pi.sections],
        'T_in [klbf (kN)]' : ['{:.1f} ({:.1f})'.format(T_in*N2lbf/1000,T_in/1000) for T_in in pi.T_inputs],
        'T_sec [klbf (kN)]': ['{:.1f} ({:.1f})'.format((T_out-T_in)*N2lbf/1000,(T_out-T_in)/1000) for T_in,T_out in zip(pi.T_inputs,pi.T_outputs)],
        'T_out [klbf (kN)]': ['{:.1f} ({:.1f})'.format(T_out*N2lbf/1000,T_out/1000) for T_out in pi.T_outputs]
    }
    indices = [i for i in range(1,len(pi.sections)+1)]
    return pd.DataFrame(data=dataDic, index=indices)

def updateGUIplots(pi):
    with outGeoSide: clear_output()
    with outGeoTop: clear_output()
    with outTension: clear_output()
    if len(pi.sections) > 0:
        with outGeoSide:
            pi.plotSideGeometry()
            if arrowCB.value: pi.plotSideArrows()
            if axisEqCB.value: plt.gca().axis('equal')
            else: plt.gca().axis('auto')
            if annotateCB.value: pi.annotateSide(annotationOffset=annotationOffsetFS.value)
            if plotCenterCB.value: pi.plotSideCenters()
            plt.gcf().set_dpi(150)
            plt.show()
        with outGeoTop:
            pi.plotTopGeometry()
            #pi.plot3dGeometry()
            if arrowCB.value: pi.plotTopArrows()
            if axisEqCB.value: plt.gca().axis('equal')
            else: plt.gca().axis('auto')
            if annotateCB.value: pi.annotateTop(annotationOffset=annotationOffsetFS.value)
            if plotCenterCB.value: pi.plotTopCenters()
            plt.gcf().set_dpi(150)
            plt.show()
        with outTension: 
            pi.plotMinTensionAlongCable()
            plt.gcf().set_dpi(150)
            plt.show()

def highlight_cells(x): #Return dataframe with styling
    df_ = x.copy()
    df_.loc[:,:] = ""
    df_.iloc[-1,-1] = "font-weight: bold"
    return df_

def styleHTMLtable(df): #https://stackoverflow.com/questions/52104682/rendering-a-pandas-dataframe-as-html-with-same-styling-as-jupyter-notebook
    styles = [
        dict(selector=" ", 
             props=[("margin","0"),
                    ("font-family",'"Helvetica", "Arial", sans-serif'),
                    ("border-collapse", "collapse"),
                    ("border","none"),
                       ]),

        #background shading
        dict(selector="tbody tr:nth-child(even)",
             props=[("background-color", "#fff")]),
        dict(selector="tbody tr:nth-child(odd)",
             props=[("background-color", "#eee")]),

        #cell spacing
        dict(selector="td", 
             props=[("padding", "0.5em"),
                    ("text-align", "center")]),
        
        dict(selector="tr:last-child td:last-child",
             props=[("font-weight", "bold")]), #Last cell in bold

        #header cell properties
        dict(selector="th", 
             props=[("font-size", "100%"),
                    ("padding", "0.5em"), #Added for space
                    ("text-align", "center")]),
    ]
    return df.style.set_table_styles(styles, overwrite=False)

def exportTable2HTML(df, filename):
    if tableOrientationRB.value==TableOrientation.VERTICAL:
        #df_styled = df.transpose().style.apply(highlight_cells,axis=None)
        df_styled = styleHTMLtable(df.transpose())
    else:
        #df_styled = df.style.apply(highlight_cells,axis=None)
        df_styled = styleHTMLtable(df)
    with open(filename, "w",encoding="utf-8") as text_file:
        text_file.write( df_styled.render() )

def create_download_link(filename, title = 'Download'):  #No idea what goes on here
    data = open(filename, 'rb').read()
    b64 = b64encode(data)
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

def suggestNextSectionInput(pi): #Suggests next angle input based on previous section
    if len(pi.sections)>0: #Check for empty list
        sec = pi.sections[-1]
        theta_out = round( np.rad2deg( -sec.theta_out if type(sec)==TopBend else sec.theta_out ) ,1) #Mirror if TopBend, convert to deg and round to one decimal point
        if (geometry_RB.value == Straight):
            if inputCoordsTypeTB.value == CoordinateSystemType.POLAR:
                angle_BFT.value = theta_out
        elif (geometry_RB.value==TopBend):
            thetaRange_FRS.value = [-theta_out, min(-theta_out+90,90)] #Suggest a bend of 90 deg
        elif (geometry_RB.value==BottomBend):
            thetaRange_FRS.value = [theta_out,  min(theta_out+90,90)] #Suggest a bend of 90 deg

def pandasTable2Matplotlib(df):
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df_copy = df.copy()
    df_copy.insert(0, '#', df_copy.index) #Insert index as column in order to see section numbers
    table = ax.table(cellText=df_copy.values, colLabels=df_copy.columns, loc='center', cellLoc='center')
    table.auto_set_column_width(col=list(range(len(df.columns))))
    table.auto_set_font_size(True)
    fig.tight_layout()

#Callback functions
def changeGeometrySpecificationUI(obj,pi):
    if geometry_RB.value == Straight:
        # secVarOptions.children = [horDisp_BFT, vertDisp_BFT]
        # sectionHBox.children = [ isWet_RB, Material_RB, geometry_RB, horDisp_BFT, vertDisp_BFT] 
        if inputCoordsTypeTB.value == CoordinateSystemType.CARTESIAN:
            secVarOptions.children = [VBox(children= [horDisp_BFT, vertDisp_BFT]),inputCoordsTypeTB]
            #secVarOptions.children = [cartesianBox, inputCoordsTypeTB]
            sectionHBox.children = [ isWet_RB, Material_RB, geometry_RB, horDisp_BFT, vertDisp_BFT] 
        else:
            secVarOptions.children = [VBox(children= [angle_BFT, length_BFT]), inputCoordsTypeTB]
            #secVarOptions.children = [polarBox, inputCoordsTypeTB]
            sectionHBox.children = [ isWet_RB, Material_RB, geometry_RB, angle_BFT, length_BFT] 
    elif geometry_RB.value == HorizontalBend:
        secVarOptions.children = [rad_BFT, psi_IS]
        sectionHBox.children = [ isWet_RB, Material_RB, geometry_RB, rad_BFT, psi_IS] 
    else:
        secVarOptions.children = [rad_BFT, thetaRange_FRS]
        sectionHBox.children = [isWet_RB, Material_RB, geometry_RB, rad_BFT, thetaRange_FRS]
    suggestNextSectionInput(pi)

def addSection_callback(obj,pi):
    s = compileSectionFromGUI(pi)
    pi.addSection(s)
    editSecBtn.disabled = False
    updateSectionOutput(pi)
    updateGUIplots(pi)
    suggestNextSectionInput(pi)

def editSection_callback(obj,pi):
    editSecBtn.disabled = True
    addSecBtn.disabled = True
    deleteScnBtn.disabled = False
    updateScnBtn.disabled = False
    #editSecNum_BIT.unobserve(adjustWidgetWithSectionSpecifyer) #Avoiding interference issues between the widgets
    obj.unobserve(functools.partial(adjustWidgetWithSectionSpecifyer,pi)) #Avoiding interference issues between the widgets
    editSecNum_BIT.max = len(pi.sections)
    editSecNum_BIT.disabled = False
    #editSecNum_BIT.observe(adjustWidgetWithSectionSpecifyer)
    obj.observe(functools.partial(adjustWidgetWithSectionSpecifyer,pi),names='value') #Avoiding interference issues between the widgets

def updateSection_callback(obj,pi):
    editSecNum_BIT.unobserve(functools.partial(adjustWidgetWithSectionSpecifyer,pi)) #To avoid observe callback function to trigger by disabling editSecNum_BIT
    for w in editList: w.disabled = True
    editSecNum_BIT.observe(functools.partial(adjustWidgetWithSectionSpecifyer,pi),names='value')
    addSecBtn.disabled = False
    editSecBtn.disabled = False
    secIndex = editSecNum_BIT.value - 1
    replacementSec = compileSectionFromGUI(pi,index=secIndex)
    pi.updateSection(replacementSec,secIndex)
    pi.updateGeometry()
    updateSectionOutput(pi)
    updateGUIplots(pi)

def deleteSection_callback(obj,pi):
    #editSecNum_BIT.unobserve(adjustwgWithSectionSpecifyer)
    for w in editList: w.disabled = True
    addSecBtn.disabled = False
    editSecBtn.disabled = False
    secIndex = editSecNum_BIT.value - 1
    pi.deleteSection(secIndex)
    pi.updateGeometry()
    updateSectionOutput(pi)
    updateGUIplots(pi)

def adjustWidgetWithSectionSpecifyer(obj,pi):
    secIndex = editSecNum_BIT.value - 1
    sec = pi.sections[secIndex]
    isWet_RB.value = sec.isWet
    Material_RB.value = sec.surfType
    geometry_RB.value = type(sec)
    if type(sec) == Straight:
        horDisp_BFT.value = sec.horDisp
        vertDisp_BFT.value = sec.vertDisp
    else:
        rad_BFT.value = sec.R
        thetaRange_FRS.value = [np.rad2deg(sec.theta_in),np.rad2deg(sec.theta_out)]
        psi_IS.value = np.rad2deg(sec.psi_out) - np.rad2deg(sec.psi_in)

def saveGlobalParameters(obj,pi):
    # global RHO_WET, RHO_DRY, MU_WET_PIPE, MU_DRY_PIPE, MU_WET_ROLLER, MU_DRY_ROLLER, T_INPUT
    # RHO_WET = rho_w_BFT.value
    # RHO_DRY = rho_d_BFT.value
    # MU_WET_PIPE = mu_wp_BFT.value
    # MU_DRY_PIPE = mu_dp_BFT.value
    # MU_WET_ROLLER = mu_wr_BFT.value
    # MU_DRY_ROLLER = mu_dr_BFT.value
    # T_INPUT = T_in_BIT.value
    globalVars = GlobalVariables(rho_wet=rho_w_BFT.value, rho_dry=rho_d_BFT.value,
                                 mu_wet_pipe=mu_wp_BFT.value, mu_dry_pipe=mu_dp_BFT.value, 
                                 mu_wet_roller=mu_wr_BFT.value, mu_dry_roller=mu_dr_BFT.value,
                                 T_input=T_in_BIT.value)
    pi.globalVars = globalVars
    saveBtn.disabled = True
    editGlbBtn.disabled = False
    for w in globalParamList: w.disabled = True
    pi.updateGlobalVariables() #Re-initializes itself
    updateGUIplots(pi)
    updateSectionOutput(pi)
    
def editGlobalParamaters(obj):
    saveBtn.disabled = False
    editGlbBtn.disabled = True
    for w in globalParamList: w.disabled = False

def exportFiles(obj,pi): #Creates one pdf of desired plots and one HTML object of table
    filenameHTML = fileNameTxt.value+'_table'+'.html'
    filenamePDF = fileNameTxt.value+'_plots'+'.pdf'
    with outPDF: clear_output()

    export2pdf(obj,pi,filenamePDF)

    df = generateDataframe(pi) if (units_RB.value==UnitType.METRIC) else generateDataframeWithImperial(pi)
    exportTable2HTML(df, filenameHTML)
    with outPDF: 
        if any([exportSidePlotCB.value,exportTopPlotCB.value,exportTensionPlotCB.value]): display(create_download_link(filenamePDF,title=filenamePDF))
        if exportTableCB.value: display(create_download_link(filename=filenameHTML, title=filenameHTML))
   
def export2separateFiles(obj,pi): #Creates files separately
    with outPDF: clear_output()

    df = generateDataframe(pi) if (units_RB.value==UnitType.METRIC) else generateDataframeWithImperial(pi)
    exportTable2HTML(df)

    pi.plotTopGeometry()
    if arrowCB.value: pi.plotTopArrows()
    if axisEqCB.value: plt.gca().axis('equal')
    else: plt.gca().axis('auto')
    plt.savefig('topGeometryPlot.png',dpi=300)

    pi.plotSideGeometry()
    if arrowCB.value: pi.plotSideArrows()
    if axisEqCB.value: plt.gca().axis('equal')
    else: plt.gca().axis('auto')
    if annotateCB.value: pi.annotateSide(annotationOffset=annotationOffsetFS.value)
    plt.savefig('sideGeometryPlot.png',dpi=300)

    pi.plotMinTensionAlongCable()
    plt.savefig('tensionPlot.png',dpi=300)

    #with outPDF: display(create_download_link(filename, title='pdf'))
    with outPDF: 
        if exportSidePlotCB.value:    display(create_download_link(filename='topGeometryPlot.png',title='topGeometry'))
        if exportTopPlotCB.value:     display(create_download_link(filename='sideGeometryPlot.png',title='sideGeometry'))
        if exportTensionPlotCB.value: display(create_download_link(filename='tensionPlot.png',title='tensionPlot'))
        if exportTableCB.value:       display(create_download_link(filename='table.html',title='Table'))

def export2pdf(obj,pi,filename): #TODO: Move to non-callback functions
    with outPDF: clear_output()
    pp = PdfPages(filename)

    if exportTopPlotCB.value:
        pi.plotTopGeometry()
        if arrowCB.value: pi.plotTopArrows()
        if axisEqCB.value: plt.gca().axis('equal')
        else: plt.gca().axis('auto')
        plt.gcf().set_dpi(300)
        pp.savefig()

    if exportSidePlotCB.value:
        pi.plotSideGeometry()
        if arrowCB.value: pi.plotSideArrows()
        if axisEqCB.value: plt.gca().axis('equal')
        else: plt.gca().axis('auto')
        if annotateCB.value: pi.annotateSide(annotationOffset=annotationOffsetFS.value)
        plt.gcf().set_dpi(300)
        pp.savefig()

    if exportTensionPlotCB.value:
        pi.plotMinTensionAlongCable()
        plt.gcf().set_dpi(300)
        pp.savefig()
    
    # if exportTableCB:
    #     df = generateDataframe(pi) if (units_RB.value==UnitType.METRIC) else generateDataframeWithImperial(pi)
    #     pandasTable2Matplotlib(df)
    #     plt.gcf().set_dpi(300)
    #     pp.savefig()

    pp.close()
    #with outPDF: display(create_download_link(filename,title=filename))

def export2csv(obj,pi):
    with outPDF: clear_output()
    df = generateDataframe(pi)
    df.to_csv('geometry.csv')
    with outPDF: display(create_download_link(filename='geometry.csv',title='geometry.csv'))

def export2excel(obj,pi):
    with outPDF: clear_output()
    df = generateDataframe(pi)
    df.to_excel('geometry.xlsx')
    with outPDF: display(create_download_link(filename='geometry.xlsx',title='geometry.xlsx'))

def axisEqualCB_callback(obj,pi):
    updateGUIplots(pi)

def arrowCB_callback(obj,pi):
    pi.plotArrows = False
    updateGUIplots(pi)

def annotateCB_callback(obj,pi):
    #pi.annotateIndex = False
    annotationOffsetFS.disabled = not(annotateCB.value) #Disable slider when checkbox not in use
    updateGUIplots(pi)

def importGeometryBtn_callback(obj):
    with outText: print('To be implemented')
    # TODO fix

def units_RB_callback(obj,pi):
    updateSectionOutput(pi)

def tableOrientationRB_callback(obj,pi):
    updateSectionOutput(pi)

def annotationOffsetFS_callback(obj,pi):
    updateGUIplots(pi)

def on_upload_change(obj):
    print('change')

def plotCenterCB_callback(obj,pi):
    updateGUIplots(pi)
