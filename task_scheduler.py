import random

# SETTINGS
DEBUG_OUTPUT = False

class TaskScheduler:
    '''
    TaskScheduler is a class that schedules tasks to be run on the data based on the policy.

    Policies: random, round-robin, annealed-sampling

    '''
    def __init__(self, args, tasks, task_lengths):
        self.epoch_counter = 0
        self.task_counter = {}
        self.tasks = tasks
        self.next_task = None
        self.args = args
        
        # Used for annealed sampling
        self.num_epochs = args.epochs
        self.task_lengths = task_lengths

        print(f"TaskScheduler initialized with tasks: {tasks} with lengths: {task_lengths}")

        assert self.args.scheduling_policy in ['random', 'round-robin', 'annealed-sampling'], f"Policy {self.args.policy} not supported."

    def annealed_sampling(self):
        alpha = 1 - 0.8 * (self.epoch_counter / self.num_epochs)
        task_probs = []
        total_length = sum(self.task_lengths)
        for task_length in self.task_lengths:
            task_probs.append(1/((task_length/total_length) ** alpha))

        if DEBUG_OUTPUT:
            print(f"Task probs: {task_probs}")
            print(f"Annealed sampling alpha: {alpha}")
        
        return random.choices(self.tasks, weights=task_probs)[0]
    
    def step_epoch(self):
        
        if self.args.force_task != '':
            self.next_task = self.args.force_task
            print(f"Forcing task {self.next_task}")
        elif self.args.scheduling_policy == 'round-robin':
            self.next_task = self.tasks[self.epoch_counter % len(self.tasks)]
            print(f"Round-robin scheduling: {self.next_task}")
        elif self.args.scheduling_policy == 'annealed-sampling':
            self.next_task = self.annealed_sampling()
            print(f"Annealed sampling: {self.next_task}")
        elif self.args.scheduling_policy == 'random':
            self.next_task = random.choice(self.tasks)
            print(f"Random scheduling: {self.next_task}")
        else:
            raise ValueError(f"Policy {self.args.scheduling_policy} not supported.")
        
        print(f"Epoch {self.epoch_counter}: Next task is {self.next_task}")
        self.epoch_counter += 1
        
    def print_task_distribution(self):
        print(f"TaskScheduler task distribution: {self.task_counter}")

    def get_next_task(self):
        self.task_counter[self.next_task] = self.task_counter.get(self.next_task, 0) + 1
        return self.next_task

    def __del__(self):
        self.print_task_distribution()