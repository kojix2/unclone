require "set"
require "./tyclone/config"
require "./tyclone/errors"
require "./tyclone/input"
require "./tyclone/sanitize"
require "./tyclone/indexing"
require "./tyclone/ffi"
require "./tyclone/kernel_result"
require "./tyclone/kernel"
require "./tyclone/phyclone_kernel"
require "./tyclone/phyclone"
require "./tyclone/vi_kernel"
require "./tyclone/result"
require "./tyclone/output"
require "./tyclone/cli"
require "./tyclone/run"

module Tyclone
  VERSION = {{ `shards version #{__DIR__}`.chomp.stringify }}
  PROGRAM = "tyclone"
  SOURCE  = "https://github.com/kojix2/tyclone"

  def self.main(args = ARGV)
    parser = CLI::Parser.new
    command = parser.parse(args)
    dispatch_command(command)
  rescue ex : CliError | OptionParser::Exception
    STDERR.puts("error: #{ex.message}")
    parser = CLI::Parser.new
    STDERR.puts(parser.to_s)
    exit 1
  rescue ex : KernelError
    STDERR.puts("error: #{ex.message}")
    exit 1
  end

  private def self.dispatch_command(command)
    case command
    when HelpCommand
      puts command.help_message
    when VersionCommand
      puts "#{PROGRAM} #{VERSION}"
    when FitViCommand,
         PhyCloneRunCommand,
         PhyCloneMapCommand,
         PhyCloneConsensusCommand,
         PhyCloneTopologyReportCommand
      Run.execute(command.config)
    end
  end
end
