i = 'info'

class Logger :
    @classmethod
    def info( cls, message ) :
        print "INFO :  " + message
        return 'something'



print Logger.info(i)
