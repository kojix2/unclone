require "set"
require "./tyclone/config"
require "./tyclone/errors"
require "./tyclone/input"
require "./tyclone/sanitize"
require "./tyclone/kernel"
require "./tyclone/indexing"
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
    config = parser.parse(args)

    case config.action
    when Action::Help
      puts config.help_message
    when Action::Version
      puts "#{PROGRAM} #{VERSION}"
    when Action::Fit
      Run.execute(config)
    end
  rescue ex : CliError | OptionParser::Exception
    STDERR.puts("error: #{ex.message}")
    parser = CLI::Parser.new
    STDERR.puts(parser.help_message)
    exit 1
  rescue ex : KernelError
    STDERR.puts("error: #{ex.message}")
    exit 1
  end
end
