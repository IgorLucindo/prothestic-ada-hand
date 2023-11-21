class CurrentObject:
    def __init__(self, dict_objects, focal_length):
        self.dict_objects = dict_objects
        self.focal_length = focal_length

        self.name = "nothing"
        self.grasp = "None"
        self.prev_grasp = "None"
        self.score = 0
        self.box = []
        self.width = 0
        self.height = 0
        self.orientation = "None"
        self.dist = 0
        self.prev_dist = 0
        self.vel = 0

        self.detected = False
        self.time = 0

    # reset current object atributes
    def resetObject(self, deltaTime, resetGraspTimer):
        self.name = "nothing"
        if self.time < resetGraspTimer: self.time += deltaTime
        else:
            self.time = 0
            self.grasp = "None"
        self.score = 0
        self.box = []
        self.dist= 0
        self.vel = 0

    # choose the object with highest score
    def setObject(self, boxes, classes, scores, deltaTime, resetGraspTimer):
        self.detected = False

        # choose current object
        self.score = 0
        for i in range(len(classes)):
            if classes[i] not in self.dict_objects:
                continue

            if scores[i] > max(self.score, 0.25):
                # set current object atributes
                self.detected = True
                self.score = scores[i]
                self.name = classes[i]
                self.box = boxes[i]

        if self.detected:
            # set current object atributes
            self.time = 0
            dict_obj = self.dict_objects[self.name]
            self.grasp = dict_obj['grasp']
            self.width = int(self.box[2]) - int(self.box[0])
            self.height = int(self.box[3]) - int(self.box[1])
            if self.width > self.height:
                self.orientation = "horizontal"
            else:
                self.orientation = "vertical"
            self.dist= dict_obj['width'] * self.focal_length / self.width
            self.vel = -(self.dist - self.prev_dist)/deltaTime
        else:
            # reset current object atributes
            self.resetObject(deltaTime, resetGraspTimer)

    # set current object previous atributes
    def setPrevObject(self):
        self.prev_grasp = self.grasp
        self.prev_dist = self.dist