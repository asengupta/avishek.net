<?php
	if (!class_exists('MarkDownTagSet', false)) {
		class MarkDownTagSet extends TagSet {
		
			function getTagSetName() {
				return 'MarkDown';
			}
		
			function getExternalPluginInfos() {
				return array(
					'name' => 'MarkDown',
					'website' => 'http://michelf.com/projects/php-markdown/'
				);
			}
		
			function isExternalPluginActive() {
				return function_exists('Markdown');
			}
		
			function isExternalPluginInstalled() {
				return is_file($this->getPluginDir().'markdown.php');
			}
		
		}
	}
	$tagset = new MarkDownTagSet();
?>