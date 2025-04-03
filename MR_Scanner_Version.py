from psychopy import gui, visual, event, core
import numpy as np
import pandas as pd
import math, random, pyxid2

scanning = True

partInfo = {'Deelnemer': '', 'Run': ''}
window = gui.DlgFromDict(dictionary = partInfo, title = 'Deelnemersinformatie')
part = int(partInfo['Deelnemer'])
run = int(partInfo['Run'])

window = visual.Window(size = [1024, 768], color = 'white', fullscr = True, units = 'pix')
event.Mouse(visible = False)

text = visual.TextStim(window, color = 'black', wrapWidth = 768)
fixation = visual.ImageStim(window, image = 'Images/Learning_Fixation.png')
boat = visual.ImageStim(window, image = 'Images/Boat.png')
target = visual.ImageStim(window, image = 'Images/Learning_Target.png')
crabDist = visual.ImageStim(window)
dot = visual.Circle(window, radius = 3, lineColor = 'red', fillColor = 'red')
cage = visual.ImageStim(window, image = 'Images/Cage.png')
laser = visual.Line(window, lineColor = 'red')
heap = visual.ImageStim(window, image = 'Images/Heap.png')
crab = visual.ImageStim(window, image = 'Images/Crab.png')
mark = visual.Line(window, lineWidth = 3, lineColor = 'red')

if run == 1:
    angleList = []
    sigmaList = []
    lengthList = []

    angle = random.randint(0, 5) * 60
    tempoAngleList = np.zeros(6)
    for index in range(6):
        tempoAngleList[index] = angle
        angle = angle + 60
        if angle == 360:
            angle = 0

    tempoAngleList = np.tile(tempoAngleList, 2)
    tempoSigmaList = np.tile([64, 64, 128, 192, 192, 128], 2)
    tempoLengthList = np.concatenate([np.zeros(6), np.ones(6)])

    for block in range(10):
        check = 0
        while check == 0:
            shuffleList = list(zip(tempoAngleList, tempoSigmaList, tempoLengthList))
            np.random.shuffle(shuffleList)
            tempoAngleList, tempoSigmaList, tempoLengthList = zip(*shuffleList)
            count = 0
            if block > 0:
                if tempoAngleList[0] == angleList[-1]:
                    count += 1
            for index in range(11):
                if tempoAngleList[index] == tempoAngleList[index+1]:
                    count += 1
            if count == 0:
                check = 1
        
        angleList = np.concatenate([angleList, tempoAngleList])
        sigmaList = np.concatenate([sigmaList, tempoSigmaList])
        lengthList = np.concatenate([lengthList, tempoLengthList])
        
    np.save('angleList.npy', angleList)
    np.save('sigmaList.npy', sigmaList)
    np.save('lengthList.npy', lengthList)
    
    columnList = ['part', 'block', 'angle', 'mu', 'sigma', 'trial', 'cage', 'crab', 'reward']
    dataFrame = pd.DataFrame(-1, index = np.arange(120 + (60 * 2) + (60 * 8) + 60), columns = columnList)
    dataFrame['part'] = part
    dataFrame['respTime'] = -1.0
    dataFrame['scanTime'] = -1.0
    dataFrame['blockTime'] = -1.0
    row = 0
    
    tempoJitterList = [3] * 12 + [3.5] * 9 + [4] * 6 + [4.5] * 3 + [5] * 3 + [5.5] * 2 + [6] * 1 + [6.5] * 1 + [7] * 1 + [7.5] * 1 + [8] * 1
    jitterList = np.zeros([3, 4, 40], dtype = float)
    countList = np.zeros([3, 4], dtype = int)
    
    for sigma in range(3):
        for point in range(4):
            np.random.shuffle(tempoJitterList)
            jitterList[sigma, point] = tempoJitterList
            
    np.save('jitterList.npy', jitterList)
    
else:
    angleList = np.load('angleList.npy')
    sigmaList = np.load('sigmaList.npy')
    lengthList = np.load('lengthList.npy')
    
    dataFrame = pd.read_csv('Data/Learning_Phase_Data_Participant_{}.csv'.format(part))
    row = dataFrame[dataFrame.block == -1].index[0]
    
    jitterList = np.load('jitterList.npy')
    countList = np.load('countList.npy')

markX = np.zeros(2)
markY = [-313, -299, -299, -313]

crabTotal = 0

respClock = core.Clock()
blockClock = core.Clock()

if scanning == False:
    devices = pyxid2.get_xid_devices()
    dev = devices[0]
    
if scanning == True:
    import scannertrigger as s
    portType = 'cedrus'
    MR_settings = {'devicenr': 0, 'sync': 4}
    st = s.ScannerTrigger.create(window, blockClock, portType, portConfig = MR_settings, timeout = 999999, esc_key = 'escape')
    st.open()
    dev = st.port
    
if run == 1:
    text.text = 'De eerste ronde zal zo dadelijk beginnen.'
elif run == 2:
    text.text = 'De tweede ronde zal zo dadelijk beginnen.'
elif run == 3:
    text.text = 'De derde ronde zal zo dadelijk beginnen.'
else:
    text.text = 'De vierde ronde zal zo dadelijk beginnen.'
text.draw()
window.flip()
startBlock = event.waitKeys(keyList = ['space'])

if scanning == True:
    try:
        text.draw()
        window.flip()
        print('Waiting for Scanner')
        triggered = st.waitForTrigger(skip = 5)
        print('Scanner OK')
    except Exception as e:
        print('Scanner Error: {0}'.format(e))
        core.quit()
    
blockClock.reset()

for block in range((run - 1) * 30, (run - 1) * 30 + 30):
    
    print('Block: {}/30'.format((block + 1) - (run - 1) * 30))
    
    sampSigma = sigmaList[block]
    priorSigma = 256 - sampSigma
    sampMu = -1024
    while sampMu < -1.64485 * priorSigma or sampMu > 1.64485 * priorSigma:
        sampMu = np.round(np.random.normal(0, priorSigma))
    
    boatX = np.round(284 * math.cos(angleList[block] * math.pi / 180))
    boatY = np.round(284 * -math.sin(angleList[block] * math.pi / 180))
    boat.pos = [boatX, boatY]
    
    dotX = np.round(80 * math.cos(angleList[block] * math.pi / 180))
    dotY = 256 + np.round(80 * -math.sin(angleList[block] * math.pi / 180))
    dot.pos = [dotX, dotY]
    
    sigma = int(sampSigma / 64 - 1)
    jitter = jitterList[sigma, 0, countList[sigma, 0]]
    countList[sigma, 0] += 1
    
    fixation.draw()
    boat.draw()
    window.flip()
    startBlock = blockClock.getTime()
    core.wait(jitter)
    
    dataFrame['block'][row] = block
    dataFrame['angle'][row] = angleList[block]
    dataFrame['scanTime'][row] = jitter
    dataFrame['blockTime'][row] = startBlock
    row += 1
    
    cageX = 0
    laserStartX = 0
    laserEndX = 0
    
    nTrials = int(lengthList[block] * 6 + 2)
    if nTrials == 8:
        heapList = np.zeros(8)
    
    for trial in range(nTrials):
        cageY = 98
        laserStartY = 62
        laserEndY = -310
        
        finish = 0
        frameCount = 0
        firstLoop = 1
        valResp = 0
        firstResp = 0
        crabList = np.zeros(5)
        
        dev.poll_for_response()
        while len(dev.response_queue):
            dev.clear_response_queue()
            dev.poll_for_response()
        
        while finish == 0:
            cage.pos = [cageX, cageY]
            laser.start = [laserStartX, laserStartY]
            laser.end = [laserEndX, laserEndY]
            
            target.draw()
            dot.draw()
            cage.draw()
            laser.draw()
            if trial > 0:
                heap.draw()
                for index in range(2):
                    mark.start = [markX[0], markY[index * 2]]
                    mark.end = [markX[1], markY[index * 2 + 1]]
                    mark.draw()
            window.flip()
            if firstLoop == 1:
                startTrial = blockClock.getTime()
                respClock.reset()
                firstLoop = 0
            
            if trial == 0:
                while valResp == 0:
                    while not dev.has_response():
                        dev.poll_for_response()
                    resp = dev.get_next_response()
                    if resp['key'] == 1:
                        respTime = respClock.getTime()
                        valResp = 1
                        finish = 1
            
            else:
                if firstResp == 0:
                    while valResp == 0:
                        while not dev.has_response():
                            dev.poll_for_response()
                        resp = dev.get_next_response()
                        if resp['key'] in [1, 2, 3]:
                            valResp = 1
                            firstResp = 1
                else:
                    dev.poll_for_response()
                    if dev.has_response():
                        resp = dev.get_next_response()
                        if resp['pressed'] == 0:
                            keyPress = 1
                
                if resp['key'] == 2 and resp['pressed'] == 0 and keyPress == 1 and cageX > -410:
                    cageX = cageX - 10
                    laserStartX = laserStartX - 10
                    laserEndX = laserEndX - 10
                    frameCount = 0
                    keyPress = 0
                elif resp['key'] == 2 and resp['pressed'] == 1 and frameCount < 10:
                    frameCount = frameCount + 1
                elif resp['key'] == 2 and resp['pressed'] == 1 and frameCount >= 10 and cageX > -410:
                    if frameCount % 2 == 0:
                        cageX = cageX - 10
                        laserStartX = laserStartX - 10
                        laserEndX = laserEndX - 10
                    frameCount = frameCount + 1
                elif resp['key'] == 3 and resp['pressed'] == 0 and keyPress == 1 and cageX < 410:
                    cageX = cageX + 10
                    laserStartX = laserStartX + 10
                    laserEndX = laserEndX + 10
                    frameCount = 0
                    keyPress = 0
                elif resp['key'] == 3 and resp['pressed'] == 1 and frameCount < 10:
                    frameCount = frameCount + 1
                elif resp['key'] == 3 and resp['pressed'] == 1 and frameCount >= 10 and cageX < 410:
                    if frameCount % 2 == 0:
                        cageX = cageX + 10
                        laserStartX = laserStartX + 10
                        laserEndX = laserEndX + 10
                    frameCount = frameCount + 1
                elif resp['key'] == 1:
                    respTime = respClock.getTime()
                    finish = 1
        
        markX[0] = cageX - 7
        markX[1] = cageX + 7
        
        crabX = -1024
        while crabX < sampMu - 1.64485 * sampSigma or crabX > sampMu + 1.64485 * sampSigma:
            crabX = np.round(np.random.normal(sampMu, sampSigma))
        heapX = crabX
        if nTrials == 8:
            heapList[trial] = heapX
        heap.pos = [heapX, -306]
        for index in range(5):
            crabList[index] = crabX
        
        for frame in range(15):
            cageY = cageY - 12.4
            cage.pos = [cageX, cageY]
            laserStartY = laserStartY - 12.4
            laser.start = [laserStartX, laserStartY]
            
            crabList[0] = crabList[0] - 6.27
            crabList[1] = crabList[1] - 3.135
            crabList[3] = crabList[3] + 3.135
            crabList[4] = crabList[4] + 6.27
            
            target.draw()
            dot.draw()
            heap.draw()
            for index in range(5):
                crab.pos = [crabList[index], -302]
                crab.draw()
            cage.draw()
            laser.draw()
            window.flip()
            
        for frame in range(15):
            cageY = cageY - 12.4
            cage.pos = [cageX, cageY]
            laserStartY = laserStartY - 12.4
            laser.start = [laserStartX, laserStartY]
            
            target.draw()
            dot.draw()
            heap.draw()
            for index in range(5):
                crab.pos = [crabList[index], -302]
                crab.draw()
            cage.draw()
            laser.draw()
            window.flip()
            
        crabCount = 0
        for index in range(5):
            if crabList[index] >= cageX - 96 and crabList[index] <= cageX + 96:
                crabCount += 1
                crabTotal += 1
        
        for frame in range(15):
            target.draw()
            dot.draw()
            heap.draw()
            for index in range(5):
                if crabList[index] >= cageX - 96 and crabList[index] <= cageX + 96:
                    crab.pos = [crabList[index], -302]
                    crab.draw()
            cage.draw()
            window.flip()
        
        if trial < 2:
            if trial == 0:
                sigma = int(sampSigma / 64 - 1)
                jitter = jitterList[sigma, 1, countList[sigma, 1]]
                countList[sigma, 1] += 1
                    
            if trial == 1:
                sigma = int(sampSigma / 64 - 1)
                jitter = jitterList[sigma, 2, countList[sigma, 2]]
                countList[sigma, 2] += 1
                
            target.draw()
            dot.draw()
            heap.draw()
            for index in range(2):
                mark.start = [markX[0], markY[index * 2]]
                mark.end = [markX[1], markY[index * 2 + 1]]
                mark.draw()
            window.flip()
            core.wait(jitter)
        
        dataFrame['block'][row] = block
        dataFrame['angle'][row] = angleList[block]
        dataFrame['mu'][row] = sampMu
        dataFrame['sigma'][row] = sampSigma
        dataFrame['trial'][row] = trial
        dataFrame['cage'][row] = cageX
        dataFrame['crab'][row] = crabX
        dataFrame['reward'][row] = crabCount
        dataFrame['respTime'][row] = respTime
        if trial < 2:
            dataFrame['scanTime'][row] = jitter
        dataFrame['blockTime'][row] = startTrial
        row += 1
        
    if nTrials == 8:

        crabDist.pos = [sampMu, -346]
        if sampSigma == 64:
            crabDist.image = 'Images/Low_Sigma_Crab_Dist.png'
        elif sampSigma == 128:
            crabDist.image = 'Images/Medium_Sigma_Crab_Dist.png'
        else:
            crabDist.image = 'Images/High_Sigma_Crab_Dist.png'
        
        sigma = int(sampSigma / 64 - 1)
        jitter = jitterList[sigma, 3, countList[sigma, 3]]
        countList[sigma, 3] += 1
        
        target.draw()
        dot.draw()
        crabDist.draw()
        for index in range(nTrials):
            heap.pos = [heapList[index], -306]
            heap.draw()
        window.flip()
        startFeedback = blockClock.getTime()
        core.wait(jitter)
        
        dataFrame['block'][row] = block
        dataFrame['angle'][row] = angleList[block]
        dataFrame['scanTime'][row] = jitter
        dataFrame['blockTime'][row] = startFeedback
        row += 1

dataFrame.to_csv('Data/Learning_Phase_Data_Participant_{}.csv'.format(part), index = False)
np.save('countList.npy', countList)

if run == 1:
    text.text = 'Je hebt het einde van de eerste ronde bereikt.\n\nIn deze ronde heb je {} krabben gevangen!'.format(crabTotal)
elif run == 2:
    text.text = 'Je hebt het einde van de tweede ronde bereikt.\n\nIn deze ronde heb je {} krabben gevangen!'.format(crabTotal)
elif run == 3:
    text.text = 'Je hebt het einde van de derde ronde bereikt.\n\nIn deze ronde heb je {} krabben gevangen!'.format(crabTotal)
else:
    text.text = ('Je hebt het einde van de vierde ronde bereikt.\n\nIn deze ronde heb je {} krabben gevangen!\n\nDe vijfde ronde zal plaats vinden rond een nieuw eiland, '
                 'waar je, in de plaats van krabben, zeepaardjes zal proberen vangen.\n\nTijdens deze ronde zullen we je telkens naar één van twee locaties rond het '
                 'eiland varen, waar jij telkens één kooi zal mogen laten vallen. Je zal de kooi echter niet kunnen verplaatsen, enkel laten vallen.'.format(crabTotal))
text.draw()
window.flip()
endBlock = event.waitKeys(keyList = ['space'])
        
if scanning == True:
    st.close()

core.quit()
