import commands

(status, output) = commands.getstatusoutput('./test.sh')

print(output)
